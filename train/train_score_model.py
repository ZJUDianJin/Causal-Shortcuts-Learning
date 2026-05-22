import json
import sys
import os
from datetime import datetime
import random
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt



def build_prompt(self, tokenizer, prompt_text, answer_text):
    messages = [{"role": "user", "content": prompt_text}]

    prompt_template_text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    prompt_ids = tokenizer.encode(prompt_template_text, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=True)
    input_ids = prompt_ids + answer_ids + [126081]
    prompt_length = len(prompt_ids)

    return input_ids, prompt_length


class TokenScoreDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        self.mask_id = 126336

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                question = item["question"]
                pred_steps = item["pred_steps"]

                messages = [{"role": "user", "content": question}]
                prompt_template_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_ids = self.tokenizer.encode(prompt_template_text, add_special_tokens=True)

                for ps in pred_steps:
                    pred_text = ps["pred"]
                    token_scores = ps["score"]

                    pred_enc = tokenizer(pred_text, add_special_tokens=False)
                    pred_ids = pred_enc["input_ids"]

                    input_ids = torch.tensor(prompt_ids + pred_ids, dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)

                    labels = torch.tensor([0] * len(prompt_ids) + token_scores, dtype=torch.float)

                    if len(labels) != len(input_ids):
                        continue

                    self.samples.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TransformerHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, mask=None):
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.out(x).squeeze(-1)


class TokenScoreModel(nn.Module):
    def __init__(self, model_name="../models/LLaDA-8B-Instruct"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=torch.float16,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder = get_peft_model(self.encoder, lora_config)
        self.encoder.print_trainable_parameters()

        hidden_size = self.encoder.config.hidden_size
        self.head = TransformerHead(hidden_size)

        device = next(self.encoder.parameters()).device
        self.head = self.head.to(device)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        hidden = outputs.hidden_states[-1]

        head_device = next(self.head.parameters()).device
        hidden = hidden.to(device=head_device, dtype=next(self.head.parameters()).dtype)

        scores = self.head(
            hidden,
            mask=(attention_mask == 0 if attention_mask is not None else None)
        )
        return scores


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def pair_rank_loss(labels, preds, mask, pairs_per_sample=20, topk_ratio=0.2,
                   local_radius=5.0, local_weight=0.3):

    device = labels.device
    mask = mask.bool().contiguous()
    mask_idx = mask.nonzero(as_tuple=True)[0]
    unmask_idx = (~mask).nonzero(as_tuple=True)[0]

    pair_loss = torch.tensor(0.0, device=device)

    if mask_idx.numel() > 1:
        topk = max(1, int(mask_idx.numel() * topk_ratio))
        sorted_idx = torch.argsort(labels[mask_idx], descending=True)
        topk_mask_idx = mask_idx[sorted_idx[:topk]]

        loss = 0.0
        pair_count = 0

        for _ in range(pairs_per_sample):
            pos_rank = random.randint(0, topk - 1)
            a = topk_mask_idx[pos_rank].item()
            b = mask_idx[random.randint(0, mask_idx.numel() - 1)].item()
            if a == b:
                continue

            label_gap = labels[a] - labels[b]
            if torch.abs(label_gap) < 1e-5:
                continue

            y = torch.sign(label_gap)
            diff = preds[a] - preds[b]
            loss += F.softplus(-y * diff)
            pair_count += 1

        pair_loss = loss / pair_count if pair_count > 0 else torch.tensor(0.0, device=device)

    total_loss = pair_loss
    return total_loss


def ndcg_at_k(preds, labels, k=10):
    device = preds.device
    topk = min(k, labels.shape[0])

    topk_indices = torch.topk(preds, k=topk).indices
    rel = labels[topk_indices]

    gains = (2 ** rel - 1).float()
    discounts = torch.log2(torch.arange(topk, device=device).float() + 2)
    dcg = (gains / discounts).sum()

    ideal_rel, _ = torch.sort(labels, descending=True)
    ideal_rel = ideal_rel[:topk]
    ideal_gains = (2 ** ideal_rel - 1).float()
    ideal_discounts = torch.log2(torch.arange(topk, device=device).float() + 2)
    idcg = (ideal_gains / ideal_discounts).sum()

    return (dcg / idcg).item() if idcg > 0 else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("../models/LLaDA-8B-Instruct")
    model = TokenScoreModel(model_name="../models/LLaDA-8B-Instruct")

    optimizer = AdamW(model.parameters(), lr=2e-5)

    data_file = "../data/..." # your data here
    dataset = TokenScoreDataset(data_file, tokenizer)

    train_size = int(0.9 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.train()

    grad_accum_steps = 16
    train_epoch = 1

    for epoch in range(train_epoch):
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for step, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            pred_scores = model(input_ids)
            mask_positions = (input_ids == 126336)

            mse_loss = ((pred_scores - labels) ** 2)[mask_positions].mean()

            B = input_ids.shape[0]
            rank_loss = 0.0
            sign_loss_total = 0.0
            valid_samples = 0

            for i in range(B):
                mask = mask_positions[i]
                lab = labels[i][mask]
                pred = pred_scores[i][mask]

                if lab.numel() < 1:
                    continue

                rank_loss += pair_rank_loss(labels[i], pred_scores[i], mask, 100)
                valid_samples += 1

            if valid_samples > 0:
                rank_loss /= valid_samples

            loss = rank_loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_bar.set_postfix({"loss": float(loss)})

    print("Training complete")


if __name__ == "__main__":
    main()
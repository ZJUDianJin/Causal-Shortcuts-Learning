import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForMaskedLM,
    AdamW, get_scheduler, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from generate import plot_contribution_heatmap, plot_entropy_heatmap_masked
from generate import compute_entropy_contribution_pos_weighted
from peft import PeftModel
from torch.utils.data import Subset


class Config:
    LOG = ""
    MODEL_PATH = "../models/LLaDA-8B-Instruct"
    TOKENIZER_PATH = "../models/LLaDA-8B-Instruct"
    TRAIN_DATA_PATH = ""
    LORA_PATH = ""
    VAL_DATA_PATH = ""
    BASE_SAVE_DIR = "../models/lora_models"
    MASK_ID = 126336
    EOS_ID = 126081
    EOT_ID = 126348
    BOS_ID = 126080
    CFG_SCALE = 0.0
    MC_NUM = 1
    DATASET_NUM = -1
    BATCH_SIZE = 1
    EPOCHS = 4
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    MAX_SEQ_LEN = 1024
    GRADIENT_ACCUMULATION_STEPS = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_STEPS = 1000
    LOG_STEPS = 100
    DTYPE = torch.bfloat16
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ]
    USE_4BIT_QUANT = False
    BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16


class PreprocessedDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "raw_input_ids": torch.tensor(sample["raw_input_ids"], dtype=torch.long),
            "prompt_lengths": torch.tensor(sample["prompt_length"], dtype=torch.long),
            "causal_input_ids": torch.tensor(sample["causal_input_ids"], dtype=torch.long),
            "causal_input_mask": torch.tensor(sample["causal_input_mask"], dtype=torch.long),
        }


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    prompt_lengths = torch.tensor([x.get("prompt_length", 0) for x in batch], dtype=torch.long)
    causal_input_mask = pad_sequence(
        [torch.tensor(x["causal_input_mask"], dtype=torch.long) for x in batch],
        batch_first=True,
        padding_value=0
    )
    causal_input_ids = pad_sequence(
        [torch.tensor(x["causal_input_ids"], dtype=torch.long) for x in batch],
        batch_first=True,
        padding_value=Config.MASK_ID
    )

    return {
        "input_ids": input_ids.to(Config.DEVICE),
        "prompt_lengths": prompt_lengths.to(Config.DEVICE),
        "causal_input_mask": causal_input_mask.to(Config.DEVICE),
        "causal_input_ids": causal_input_ids.to(Config.DEVICE)
    }


def get_logits(model, batch, prompt_mask, cfg_scale, mask_id):
    if cfg_scale > 0.:
        un_batch = batch.clone()
        un_batch[prompt_mask] = mask_id
        batch = torch.cat([batch, un_batch])

    logits = model(input_ids=batch).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

    return logits


def forward_process(batch, prompt_lengths, mask_id):
    b, l = batch.shape
    prompt_mask = torch.arange(l, device=batch.device).expand(b, l) < prompt_lengths.unsqueeze(1)
    target_len = l - prompt_lengths.max().item()

    k = torch.randint(1, target_len + 1, (), device=batch.device)
    x = torch.round(
        torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
    ).long()
    x = ((x - 1) % target_len) + 1

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)

    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    pad_len = l - target_len - (prompt_lengths - prompt_lengths.min()).max().item()
    is_mask = torch.cat([
        torch.zeros(b, prompt_lengths.max().item(), dtype=torch.bool, device=batch.device),
        is_mask,
        torch.zeros(b, pad_len, dtype=torch.bool, device=batch.device)
    ], dim=1)[:, :l]

    noisy_batch = torch.where(is_mask, mask_id, batch)
    p_mask = (x / target_len).unsqueeze(1).repeat(1, l)

    return noisy_batch, is_mask, p_mask


def mask_all(batch, prompt_lengths, mask_id):
    b, l = batch.shape
    device = batch.device

    positions = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
    prompt_mask = positions < prompt_lengths.unsqueeze(1)
    is_mask = ~prompt_mask

    noisy_batch = torch.where(is_mask, mask_id, batch)
    p_mask = is_mask.float()

    return noisy_batch, is_mask, p_mask


def gift_forward_process(batch, prompt_lengths, mask_id, t0_logit):
    b, l = batch.shape
    device = batch.device
    eps = 1e-8

    prompt_mask = torch.arange(l, device=device).expand(b, l) < prompt_lengths.unsqueeze(1)
    answer_mask = ~prompt_mask

    p = torch.softmax(t0_logit, dim=-1)
    H = -torch.sum(p * torch.log(p + eps), dim=-1)

    beta = torch.sqrt(torch.clamp(H, min=eps))

    answer_beta = beta * answer_mask.float()
    sum_beta = answer_beta.sum(dim=1, keepdim=True)
    n_answer = answer_mask.sum(dim=1, keepdim=True).clamp(min=1)
    beta_ref = sum_beta / n_answer

    t = torch.rand(1, device=device).item()

    gamma = beta / beta_ref
    mask_prob = 1 - torch.pow(1.0 - t, gamma)

    mask_prob = torch.where(answer_mask, mask_prob, 0.0)
    mask_prob = mask_prob.clamp(0.0, 1.0)

    is_mask = torch.rand_like(mask_prob) < mask_prob
    noisy_batch = torch.where(is_mask, mask_id, batch)

    return noisy_batch, is_mask, mask_prob


def pad_noisy_batch(noisy_batch, is_mask, p_mask, causal_input_mask, tokenizer, max_seq_len):
    device = noisy_batch.device
    B, L = noisy_batch.shape

    input_ids = tokenizer.pad(
        {"input_ids": noisy_batch},
        padding="max_length" if L < max_seq_len else "do_not_pad",
        max_length=max_seq_len,
        return_tensors="pt"
    )["input_ids"].to(device)

    mask_index = torch.zeros((B, max_seq_len), dtype=torch.bool, device=device)
    mask_index[:, :L] = is_mask[:, :L]

    p_mask_pad = torch.zeros((B, max_seq_len), dtype=p_mask.dtype, device=device)
    p_mask_pad[:, :L] = p_mask[:, :L]

    causal_input_mask_pad = torch.nn.functional.pad(
        causal_input_mask,
        (0, max_seq_len - causal_input_mask.shape[1]),
        value=0
    )

    return input_ids, mask_index, p_mask_pad, causal_input_mask_pad


def plot_loss_curve(loss_list, save_path, title):
    if len(loss_list) == 0:
        return
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Batch")
    plt.ylabel("Average Token Loss")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def train():
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    SAVE_DIR = os.path.join(Config.BASE_SAVE_DIR, timestamp, "weight")
    LOG_DIR = os.path.join(Config.BASE_SAVE_DIR, timestamp, "logs")
    CONFIG_DIR = os.path.join(Config.BASE_SAVE_DIR, timestamp, "config")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    with open(os.path.join(CONFIG_DIR, "config.json"), "w", encoding="utf-8") as f:
        config_dict = {}
        for k, v in Config.__dict__.items():
            if not k.startswith("__"):
                config_dict[k] = str(v) if isinstance(v, (torch.device, torch.dtype)) else v
        json.dump(config_dict, f, indent=4)

    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=Config.DTYPE,
        device_map="auto"
    )

    if Config.LORA_PATH != "":
        model = PeftModel.from_pretrained(model, Config.LORA_PATH, torch_dtype=Config.DTYPE)

    model = prepare_model_for_kbit_training(model) if Config.USE_4BIT_QUANT else model

    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.train()

    train_dataset = PreprocessedDataset(Config.TRAIN_DATA_PATH)

    if Config.DATASET_NUM > 0:
        train_dataset = Subset(train_dataset, range(Config.DATASET_NUM))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_dataloader) * Config.EPOCHS

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=num_training_steps
    )

    loss_list = []
    global_step = 0
    accum_loss = 0

    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        progress_bar = tqdm(train_dataloader)

        epoch_loss_sum = 0.0
        epoch_token_count = 0

        for batch in progress_bar:
            input_ids = batch["input_ids"]
            prompt_lengths = batch["prompt_lengths"]
            causal_input_mask = batch["causal_input_mask"]

            loss = 0

            for _ in range(Config.MC_NUM):
                noisy_batch, mask_index, p_mask = forward_process(input_ids, prompt_lengths, Config.MASK_ID)

                prompt_mask = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
                    input_ids.shape[0], input_ids.shape[1]
                ) < prompt_lengths.unsqueeze(1)

                noisy_batch, mask_index, p_mask, causal_input_mask = pad_noisy_batch(
                    noisy_batch, mask_index, p_mask, causal_input_mask, tokenizer, Config.MAX_SEQ_LEN
                )

                logits = get_logits(model, noisy_batch, prompt_mask, Config.CFG_SCALE, Config.MASK_ID)

                alpha = 0.25
                beta = 1.0

                u = F.cross_entropy(logits[mask_index], input_ids[mask_index], reduction="none")
                v = alpha * (1 - torch.exp(-u)) * beta

                token_loss = (
                    F.cross_entropy(logits[mask_index], input_ids[mask_index], reduction="none")
                    / p_mask[mask_index]
                    * v
                )

                epoch_loss_sum += token_loss.sum().item()
                epoch_token_count += token_loss.numel()
                loss += token_loss.sum()

            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (global_step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                accum_loss += loss.item()
                loss_list.append(min(accum_loss, 3))
                accum_loss = 0

                plot_loss_curve(loss_list, os.path.join(CONFIG_DIR, "loss.png"), "Loss")

            global_step += 1
            progress_bar.set_postfix({"loss": loss.item()})

        save_path = os.path.join(SAVE_DIR, f"epoch_{epoch}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print("saved:", save_path)


if __name__ == "__main__":
    train()
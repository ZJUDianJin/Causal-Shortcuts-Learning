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
from typing import Tuple
import random

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
    causal_input_mask = pad_sequence([torch.tensor(x["causal_input_mask"], dtype=torch.long) for x in batch], batch_first=True, padding_value=0)
    causal_input_ids = pad_sequence([torch.tensor(x["causal_input_ids"], dtype=torch.long) for x in batch], batch_first=True, padding_value=Config.MASK_ID)
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
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
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

def pad_noisy_batch(noisy_batch, is_mask, p_mask, causal_input_mask, tokenizer, max_seq_len):
    device = noisy_batch.device
    B, L = noisy_batch.shape

    input_ids = tokenizer.pad(
        {"input_ids": noisy_batch},
        padding="max_length" if L < max_seq_len else "do_not_pad",
        max_length=max_seq_len,
        return_tensors="pt"
    )["input_ids"]

    input_ids = input_ids.to(device)

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

def two_phase_mask_at_block(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    blk_size: int,
    mask_id: int,
    prefix_mask_rate: float = 0.0,
    suffix_mask_prob: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    b, L = input_ids.shape
    device = input_ids.device

    noisy_batch = input_ids.clone()
    mask_flags = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

    for i in range(b):
        ids = input_ids[i].cpu().tolist()
        plen = prompt_lengths[i].item()

        ans_s = plen
        ans_e = L
        answer_len = ans_e - ans_s

        if answer_len <= 0:
            continue

        block_num = (answer_len + blk_size - 1) // blk_size
        t = random.random()
        blk_idx = min(int(t * block_num), block_num - 1)

        s0 = ans_s + blk_idx * blk_size
        e0 = min(ans_s + (blk_idx + 1) * blk_size, ans_e)

        noisy = ids.copy()
        mflag = [False] * L

        if ans_s < ans_e:
            prefix_s = ans_s
            prefix_e = max(min(s0, ans_e), ans_s)

            if prefix_mask_rate > 0.0 and prefix_e > prefix_s:
                for pos in range(prefix_s, prefix_e):
                    if random.random() < prefix_mask_rate:
                        noisy[pos] = mask_id

        if s0 < ans_e:
            p = random.uniform(0.001, 1.0)

            for pos in range(max(s0, ans_s), e0):
                if random.random() < p:
                    noisy[pos] = mask_id
                    mflag[pos] = True

        if suffix_mask_prob >= 1.0 - 1e-12:
            for pos in range(e0, ans_e):
                noisy[pos] = mask_id

        elif suffix_mask_prob > 1e-12:
            for pos in range(e0, ans_e):
                if random.random() < suffix_mask_prob:
                    noisy[pos] = mask_id

        noisy_batch[i] = torch.tensor(noisy, device=device)
        mask_flags[i] = torch.tensor(mflag, dtype=torch.bool, device=device)

    return noisy_batch, mask_flags

def plot_loss_curve(loss_list, save_path, title):
    if len(loss_list) == 0:
        print(f"{title} is empty, skip plotting.")
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

    print(f"Model save path: {SAVE_DIR}")
    print(f"Log save path: {LOG_DIR}")
    print(f"Config save path: {CONFIG_DIR}")

    with open(os.path.join(CONFIG_DIR, "config.json"), "w", encoding="utf-8") as f:
        config_dict = {}

        for k, v in Config.__dict__.items():
            if not k.startswith("__"):
                if isinstance(v, (torch.device, torch.dtype)):
                    config_dict[k] = str(v)
                else:
                    config_dict[k] = v

        json.dump(config_dict, f, indent=4)

    bnb_config = None

    if Config.USE_4BIT_QUANT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE
        )

    tokenizer = AutoTokenizer.from_pretrained(
        Config.TOKENIZER_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=Config.DTYPE,
        device_map="auto",
        quantization_config=bnb_config
    )

    if Config.LORA_PATH != "":
        model = PeftModel.from_pretrained(
            model,
            Config.LORA_PATH,
            torch_dtype=Config.DTYPE
        )

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

    print(len(train_dataset))

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
        print(f"===== Epoch {epoch+1}/{Config.EPOCHS} =====")

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        epoch_loss_sum = 0.0
        epoch_token_count = 0

        for batch in progress_bar:
            input_ids = batch["input_ids"]
            prompt_lengths = batch["prompt_lengths"]
            causal_input_ids = batch["causal_input_ids"]
            causal_input_mask = batch["causal_input_mask"]

            batch_dlm_loss_sum = 0.0
            batch_dlm_token_count = 0

            loss = 0

            for mc in range(Config.MC_NUM):
                blk_size = 32

                two_phase_mask_at_block(
                    input_ids,
                    prompt_lengths,
                    blk_size,
                    Config.MASK_ID
                )

                noisy_batch, mask_index, p_mask = forward_process(
                    input_ids,
                    prompt_lengths,
                    Config.MASK_ID
                )

                prompt_mask = (
                    torch.arange(
                        input_ids.shape[1],
                        device=input_ids.device
                    ).expand(
                        input_ids.shape[0],
                        input_ids.shape[1]
                    )
                    < prompt_lengths.unsqueeze(1)
                )

                noisy_batch, mask_index, p_mask, causal_input_mask = (
                    pad_noisy_batch(
                        noisy_batch,
                        mask_index,
                        p_mask,
                        causal_input_mask,
                        tokenizer,
                        Config.MAX_SEQ_LEN
                    )
                )

                logits = get_logits(
                    model,
                    noisy_batch,
                    prompt_mask,
                    Config.CFG_SCALE,
                    Config.MASK_ID
                )

                token_loss = F.cross_entropy(
                    logits[mask_index],
                    input_ids[mask_index],
                    reduction="none"
                )

                batch_dlm_loss_sum += token_loss.sum().item()
                batch_dlm_token_count += token_loss.numel()

                epoch_loss_sum += token_loss.sum().item()
                epoch_token_count += token_loss.numel()

                loss += token_loss.sum()

                if batch_dlm_token_count > 0:
                    avg_dlm_loss = batch_dlm_loss_sum / batch_dlm_token_count
                    accum_loss += avg_dlm_loss

            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (global_step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loss_list.append(
                    min(
                        accum_loss / Config.GRADIENT_ACCUMULATION_STEPS,
                        3
                    )
                )

                accum_loss = 0

                plot_loss_curve(
                    loss_list,
                    os.path.join(CONFIG_DIR, "loss.png"),
                    "DLM Average Token Loss"
                )

            global_step += 1

            progress_bar.set_postfix({
                "loss": loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
            })

        save_path = os.path.join(SAVE_DIR, f"epoch_{epoch}")

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"LoRA adapter saved to {save_path}")

        avg_epoch_loss = (
            epoch_loss_sum / epoch_token_count
            if epoch_token_count > 0 else 0
        )

        epoch_log = (
            f"Epoch {epoch+1} Finished | "
            f"Average Token Loss: {avg_epoch_loss:.6f}"
        )

        print(epoch_log)

        with open(os.path.join(LOG_DIR, "train_log.txt"), "a") as f:
            f.write(epoch_log + "\n")

    final_save_path = os.path.join(
        SAVE_DIR,
        "final_lora_model"
    )

    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    print(
        f"Training finished! "
        f"Final LoRA adapter saved to {final_save_path}"
    )

if __name__ == "__main__":
    train()
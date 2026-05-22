import os
import json
from tqdm import tqdm
import torch
import pandas as pd
import torch.multiprocessing as mp
import random

from train import SFTDataset, Config
from transformers import AutoTokenizer
from score_model import TokenScoreModel

DATA_PATH = ""
SAVE_DIR = ""
MODEL_PATH = Config.MODEL_PATH
MAX_SEQ_LEN = Config.MAX_SEQ_LEN
CI_SCORE_MODEL = "../models/CI-Score-Model"
BASE_MODEL_NAME = "../models/LLaDA-8B-Instruct"


def preprocess_worker(gpu_id, start_idx, end_idx):
    """GPU worker that processes a range of samples."""
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Using device: {device}, processing samples {start_idx}-{end_idx}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    score_model = TokenScoreModel(BASE_MODEL_NAME)
    state_dict = torch.load(f"{CI_SCORE_MODEL}/pytorch_model.bin", map_location="cpu")
    score_model.load_state_dict(state_dict, strict=False)
    score_model.to(device)
    score_model.eval()

    dataset = SFTDataset(DATA_PATH, tokenizer, MAX_SEQ_LEN, score_model)
    save_path = os.path.join(SAVE_DIR, f"train_data_gpu{gpu_id}.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(start_idx, end_idx), desc=f"GPU {gpu_id}"):
            sample = dataset[i]

            if sample is None:
                continue

            item = {
                "input_ids": sample["input_ids"].tolist(),
                "raw_input_ids": sample["raw_input_ids"].tolist(),
                "prompt_length": int(sample["prompt_length"]),
                "causal_input_mask": sample["causal_input_mask"].tolist(),
                "causal_input_ids": sample["causal_input_ids"].tolist(),
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

            if (i - start_idx + 1) % 100 == 0:
                f.flush()

    print(f"[GPU {gpu_id}] Finished, saved to {save_path}")


def main():
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs, splitting data ...")

    df = pd.read_parquet(DATA_PATH)
    all_data = df.to_dict(orient="records")

    random.seed(42)
    random.shuffle(all_data)

    total_samples = len(all_data)
    split_size = total_samples // n_gpus

    ranges = [(i * split_size, (i + 1) * split_size) for i in range(n_gpus - 1)]
    ranges.append(((n_gpus - 1) * split_size, total_samples))

    processes = []
    for gpu_id, (start, end) in enumerate(ranges):
        p = mp.Process(target=preprocess_worker, args=(gpu_id, start, end))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
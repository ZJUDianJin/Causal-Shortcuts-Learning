import torch
import sys
import os
import numpy as np
import torch.nn.functional as F
from score_model import TokenScoreModel
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    """
    Gumbel max sampling for categorical distributions.
    Low precision may affect generation quality.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps,
        device=mask_index.device,
        dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def compute_entropy_contribution(model, x, ans, mask_id=126336, temperature=0., attention_mask=None):
    L = x.shape[1]
    contributions = torch.zeros(L, device=x.device)

    logits = model(x, attention_mask=attention_mask).logits
    probs = F.softmax(logits, dim=-1)
    H0 = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).squeeze(0)

    mask_indices = (x == mask_id).nonzero(as_tuple=True)[1]

    for idx in mask_indices:
        x_ = x.clone()

        logit_idx = logits[0, idx]
        if temperature > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(logit_idx)))
            token_id = torch.argmax(logit_idx + gumbel)
        else:
            token_id = torch.argmax(logit_idx)

        x_[:, idx] = ans[:, idx]

        logits_new = model(x_, attention_mask=attention_mask).logits
        probs_new = F.softmax(logits_new, dim=-1)
        H_new = -(probs_new * torch.log(probs_new + 1e-12)).sum(dim=-1).squeeze(0)

        H_change = H0 - H_new
        H_change[idx] = 0

        remain_mask = (x_ == mask_id).nonzero(as_tuple=True)[1]
        contributions[idx] = H0[remain_mask].mean() - H_new[remain_mask].mean()

    return contributions


def compute_entropy_contribution_weighted(model, x, ans, mask_id=126336, temperature=0., attention_mask=None, sigma=None):
    L = x.shape[1]
    contributions = torch.zeros(L, device=x.device)
    sigma = sigma or (L / 4)

    logits = model(x, attention_mask=attention_mask).logits
    probs = F.softmax(logits, dim=-1)
    H0 = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).squeeze(0)

    mask_indices = (x == mask_id).nonzero(as_tuple=True)[1]

    for idx in mask_indices:
        x_ = x.clone()
        x_[:, idx] = ans[:, idx]

        logits_new = model(x_, attention_mask=attention_mask).logits
        probs_new = F.softmax(logits_new, dim=-1)
        H_new = -(probs_new * torch.log(probs_new + 1e-12)).sum(dim=-1).squeeze(0)

        remain_mask = (x_ == mask_id).nonzero(as_tuple=True)[1]

        if len(remain_mask) == 0:
            contributions[idx] = 0
            continue

        dist = (remain_mask - idx).float()
        weights = torch.exp(-(dist ** 2) / (2 * sigma ** 2))

        H_diff = H0[remain_mask] - H_new[remain_mask]
        contributions[idx] = (H_diff * weights).sum() / weights.sum()

    return contributions


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False
):
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length),
                       dtype=attention_mask.dtype,
                       device=model.device)
        ], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length:
               prompt.shape[1] + (num_block + 1) * block_length:] == mask_id
        )

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)

                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True
    )

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    assert tokenizer.pad_token_id != 126336

    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
    ]

    messages = [{"role": "user", "content": prompt} for prompt in prompts]

    prompts = [
        tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False)
        for m in messages
    ]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )

    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(
        model,
        input_ids,
        attention_mask,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.,
        cfg_scale=0.,
        remasking='low_confidence'
    )

    output = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:],
        skip_special_tokens=True
    )

    for o in output:
        print(o)
        print('-' * 50)


if __name__ == '__main__':
    main()
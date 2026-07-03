from .logger import RougeEvalLogger

from tqdm import tqdm
from tqdm.contrib import tzip
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

class ListDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def eval(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128,
    batch_size: int = 8
):
    logger = RougeEvalLogger()

    if tokenizer.pad_token is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    original_padding_side = tokenizer.padding_side

    tokenizer.padding_side = 'left'

    dataset = ListDataset(list(zip(prompts, gts)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model_max_length = getattr(model.config, "max_position_embeddings", 2048)

    for batch in tqdm(loader, desc="VerbMem"):
        batch_prompts, batch_gts = batch

        inputs = tokenizer(
            list(batch_prompts), 
            return_tensors='pt',
            add_special_tokens=True, 
            padding=True, 
            truncation=True
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # モデルの最大長(1024)から入力長を引いて、安全に生成できるトークン数を計算
        input_length = input_ids.shape[1]
        safe_max_new_tokens = min(max_new_tokens, model_max_length - input_length)
        # --------------------------------------------------

        decoded_preds = []
        if safe_max_new_tokens > 0:
            with torch.no_grad():
                # Use the `model` to generate the continuation of the `input_ids`.
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask, # attention_maskを渡す
                    max_new_tokens=safe_max_new_tokens, # 安全な値を渡す
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            generated_ids = output_ids[:, input_ids.shape[-1]:]
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        else:
            decoded_preds = [""] * len(batch_prompts)
        
        # GT Handling (Batch processing)
        tokenizer.padding_side = 'right' 
        gt_inputs = tokenizer(list(batch_gts), return_tensors='pt', padding=True, truncation=True)
        gt_ids = gt_inputs.input_ids[:, :max_new_tokens]
        gt_shorts = tokenizer.batch_decode(gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tokenizer.padding_side = 'left' # Restore

        for p, gt_short, out in zip(batch_prompts, gt_shorts, decoded_preds):
            logger.log(p, gt_short, out)

    tokenizer.padding_side = original_padding_side
    return logger.report()

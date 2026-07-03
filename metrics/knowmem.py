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

    
def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string

def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32,
    batch_size: int = 16
):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    # Few-shot prompting
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\\nAnswer: {answer}\\n\\n"

    full_prompts = [
        general_prompt + f"Question: {question}\nAnswer: " for question in questions
    ]

    # Setup Padding
    if tokenizer.pad_token is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    original_padding_side = tokenizer.padding_side

    tokenizer.padding_side = 'left' 

    dataset = ListDataset(list(zip(full_prompts, answers, questions)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model_max_length = getattr(model.config, "max_position_embeddings", 2048)

    for batch in tqdm(loader, desc="KnowMem"):
        batch_prompts, batch_answers, batch_questions = batch

        # Encode the `prompt` into `input_ids`
        inputs = tokenizer(
            list(batch_prompts), 
            return_tensors='pt', 
            add_special_tokens=True,
            padding=True, 
            truncation=True
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # モデルの最大長から入力長を引いて、安全に生成できるトークン数を計算
        input_length = input_ids.shape[1]
        model_max_length = model.config.max_position_embeddings
        safe_max_new_tokens = min(max_new_tokens, model_max_length - input_length)
        # --------------------------------------------------

        if safe_max_new_tokens <= 0:
            # Fallback if input is already too long
            decoded_preds = [""] * len(batch_prompts)
        else:
            with torch.no_grad():
                # Use the `model` to generate the continuation of the `input_ids`.
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=safe_max_new_tokens, # 安全な値を渡す
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Extract generated tokens
            generated_ids = output_ids[:, input_ids.shape[-1]:]
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Post-processing
        for q_text, gt, output in zip(batch_questions, batch_answers, decoded_preds):
            final_output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
            logger.log(f"Question: {q_text}", gt, final_output)

    tokenizer.padding_side = original_padding_side
    return logger.report()

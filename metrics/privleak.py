import sys
sys.path.append(".")
sys.path.append("../baselines")

from typing import List, Dict
import torch
from tqdm import tqdm
import zlib
import numpy as np
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve
from torch.utils.data import DataLoader, Dataset

class ListDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    
def compute_ppl(texts: List[str], model, tokenizer, device='cuda'):
    # Setup Padding
    if tokenizer.pad_token is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    original_padding_side = tokenizer.padding_side

    tokenizer.padding_side = 'right' 
    # ----------------------------------------------

    # model.config から最大長を取得する
    model_max_length = model.config.max_position_embeddings
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=model_max_length)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) # attention maskの取得
    
    # Label is input_ids, but masked where it is padding
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
    
    # Average loss of batch (not used for individual scores but good for debug)
    logits = outputs.logits
    loss_val = outputs.loss.item() 

    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # モデル出力と同じデバイスにラベルを移動
    shift_labels = shift_labels.to(shift_logits.device)
    # ------------------------------------------------
    
    # Token-wise Cross Entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size()) # [batch, seq_len]

    token_losses_np = token_losses.detach().float().cpu().numpy()
    shift_labels_np = shift_labels.cpu().numpy()
    
    batch_results = []
    
    for i in range(len(texts)):
        # Extract valid token losses for this sample
        mask = shift_labels_np[i] != -100
        valid_losses = token_losses_np[i][mask]
        
        # log_prob = -loss
        all_prob = (-valid_losses).tolist()
        
        # PPL = exp(mean(loss))
        if len(valid_losses) > 0:
            mean_loss = valid_losses.mean().item()
            ppl = np.exp(mean_loss)
            total_loss = mean_loss # Using mean loss as the 'loss' metric compatible with original
        else:
            ppl = 0.0
            total_loss = 0.0
            
        batch_results.append({
            'ppl': ppl,
            'all_prob': all_prob,
            'loss': total_loss
        })
        
    tokenizer.padding_side = original_padding_side
    return batch_results


def inference(res_original, res_lower, zlib_entropy) -> Dict:
    pred = {}

    p1_likelihood = res_original['loss']
    p_lower_likelihood = res_lower['loss']

    pred["PPL"] = float(p1_likelihood)
    # p_lower_likelihoodが0になるのを防ぐ
    pred["PPL/lower"] = float(p1_likelihood / (p_lower_likelihood + 1e-6))
    pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)

    # min-k prob
    all_prob = res_original['all_prob']
    if all_prob: # all_probが空でないことを確認する
        for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            k_length = int(len(all_prob)*ratio)
            topk_prob = np.sort(all_prob)[:k_length]
            pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())

    return pred


def eval_data(data: List[str], model, tokenizer, batch_size=8):
    out = []
    # Pre-calculate zlib entropy (CPU bound)
    zlib_entropies = [len(zlib.compress(bytes(t, 'utf-8'))) for t in data]
    
    dataset = ListDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    global_idx = 0
    
    for batch_texts in tqdm(loader, desc="PrivLeak Batch"):
        # Original
        batch_res = compute_ppl(batch_texts, model, tokenizer, device=model.device)
        
        # Lowercase
        batch_texts_lower = [t.lower() for t in batch_texts]
        batch_res_lower = compute_ppl(batch_texts_lower, model, tokenizer, device=model.device)
        
        # Aggregate results
        for i in range(len(batch_texts)):
            res = inference(batch_res[i], batch_res_lower[i], zlib_entropies[global_idx])
            out.append({'text': batch_texts[i]} | res)
            global_idx += 1
    return out


def sweep(ppl, y):
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, get_auc(fpr, tpr), acc


def eval(
    forget_data: List[str],
    retain_data: List[str],
    holdout_data: List[str],
    model, tokenizer,
    batch_size: int = 8
):
    log = {}
    print("Evaluating on the forget set...")
    log['forget'] = eval_data(forget_data, model, tokenizer, batch_size)
    print("Evaluating on the retain set...")
    log['retain'] = eval_data(retain_data, model, tokenizer, batch_size)
    print("Evaluating on the holdout set...")
    log['holdout'] = eval_data(holdout_data, model, tokenizer, batch_size)

    auc = {}

    ppl_types = [
        "PPL", "PPL/lower", "PPL/zlib",
        "Min-5%", "Min-10%", "Min-20%", "Min-30%",
        "Min-40%", "Min-50%", "Min-60%"
    ]

    for split0 in ['forget', 'retain', 'holdout']:
        for split1 in ['forget', 'retain', 'holdout']:
            if not log[split0] or not log[split1]: continue
            log0, log1 = log[split0], log[split1]
            for ppl_type in ppl_types:
                ppl_nonmember = [d.get(ppl_type, 0) for d in log0]
                ppl_member = [d.get(ppl_type, 0) for d in log1]
                ppl = np.array(ppl_nonmember + ppl_member)
                y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                if len(np.unique(y)) < 2: continue

                try:
                    _, _, auc_score, _ = sweep(ppl, y)
                    auc[f"{split0}_{split1}_{ppl_type}"] = auc_score
                except:
                    auc[f"{split0}_{split1}_{ppl_type}"] = 0.0

    return auc, log

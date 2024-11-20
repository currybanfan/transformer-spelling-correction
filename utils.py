import random
import torch
import Levenshtein
from torch import nn

def levenshtein_distance(pred_str, target_str):
    return Levenshtein.distance(pred_str, target_str)

def compute_levenshtein_loss(pred_str_batch, target_str_batch):

    total_distance = 0
    for pred, target in zip(pred_str_batch, target_str_batch):
        total_distance += levenshtein_distance(pred, target)
    
    # 計算平均損失
    avg_distance = total_distance / len(pred_str_batch)
    return avg_distance

def compute_length_loss(pred, target_ids, eos_idx):
    # 定義函數來獲取 [eos] 的位置
    def get_eos_length(sequence, eos_idx):
        eos_positions = (sequence == eos_idx).nonzero(as_tuple=True)
        if eos_positions[0].numel() > 0:  # 確認是否找到 [eos]
            return eos_positions[0][0].item()  # 如果找到，返回第一個 [eos] 的位置
        else:
            return sequence.size(0)  # 若未找到 [eos]，則返回整個序列長度

    # 使用 get_eos_length 計算 target 和 pred 的長度
    target_lengths = torch.tensor([get_eos_length(seq, eos_idx) for seq in target_ids], device=target_ids.device)
    pred_lengths = torch.tensor([get_eos_length(seq, eos_idx) for seq in pred], device=pred.device)

    # 使用 MSELoss 計算長度差異損失
    length_loss = nn.MSELoss()(pred_lengths.float(), target_lengths.float())
    return length_loss

def metrics(pred:list, target:list) -> float:
    """
    pred: list of strings
    target: list of strings

    return: accuracy(%)
    """
    if len(pred) != len(target):
        raise ValueError('length of pred and target must be the same')
    correct = 0
    for i in range(len(pred)):
        if pred[i] == target[i]:
            correct += 1
    return correct / len(pred) * 100

def gen_padding_mask(src, pad_idx):
    # detect where the padding value is
    return (src == pad_idx)

def get_index(pred, dim=1):
    return pred.clone().argmax(dim=dim)

def random_change_idx(data: torch.Tensor, prob, pad_idx):
    # randomly change the index of the input data
    for word in data:
         for i in range(1, len(word) - 1):
            if word[i].item() == pad_idx:  # 若遇到 pad_idx 則跳過處理
                break
            if random.random() < prob:  # 機率決定是否改變索引
                word[i] = random.randint(0, 25)
    return data


def random_masked(data: torch.Tensor, prob, mask_idx, pad_idx):
    # randomly mask the input data
    for word in data:
        for i in range(1, len(word) - 1):
            if word[i].item() == pad_idx:  # 若遇到 pad_idx 則跳過處理
                break
            if random.random() < prob:  # 機率決定是否遮罩索引
                word[i] = mask_idx
    return data

def random_mask_target(tgt_ids, sampling_prob, pad_idx, sos_idx):
    """遮蔽目標單字的 token id"""
    mask = torch.rand(tgt_ids.size(), device=tgt_ids.device) > sampling_prob
    tgt_ids[mask] = pad_idx
    tgt_ids[:, 0] = sos_idx  # 確保 [sos] 保留在序列的起始位置
    return tgt_ids
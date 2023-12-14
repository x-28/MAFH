import torch
import numpy as np

def quantizationLoss(hashrepresentations_bss, hashcodes_bs):
    for index, hashrepresentations_bs in enumerate(hashrepresentations_bss):
        batch_size, bit = hashcodes_bs.shape
        quantization_loss = torch.sum(torch.pow(hashcodes_bs - hashrepresentations_bs, 2)) / (batch_size * bit)
        quantization_loss += quantization_loss
    return quantization_loss/len(hashrepresentations_bss)



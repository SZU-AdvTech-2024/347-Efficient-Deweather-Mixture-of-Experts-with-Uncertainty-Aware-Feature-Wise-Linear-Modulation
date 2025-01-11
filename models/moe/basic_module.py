import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions.normal import Normal

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

####################################  Top K  ###############################

class KeepTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # x: [b, n, d]
        if self.top_k == 0:
            return x
        filter_value=-float('Inf')
        indices_to_remove = x < torch.topk(x, self.top_k)[0][..., -1, None]  # topk返回value的最内层大小比较
        x[indices_to_remove] = filter_value
        return x
    

class DynamicTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k
        self.top_p = 0.6

    def forward(self, x):
        # x: [bs, np, ne]
        filter_value=-float('Inf')

        # logits = torch.tensor([[[0.3,0.4,0.1],
        #                  [0.1,0.1,0.0]],
        #                  [[0.1,0.3,0.4],
        #                   [0.9, 0.0, 0.1]]])

        logits = torch.softmax(x, dim=-1)

        # print(f"logits is {logits}.")

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_logits = torch.cumsum(sorted_logits, dim=-1)
        mask = cumulative_logits > self.top_p
        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask
        
        top_k_mask = torch.zeros_like(mask)
        top_k_mask[..., self.top_k:] = True
        mask = mask | top_k_mask.bool()


        sorted_logits = torch.where(mask, filter_value, sorted_logits)
        logits = torch.gather(sorted_logits, dim=-1, index=sorted_indices)

        # print(f"now, logits is {logits}.")
        # exit(0)

        return logits

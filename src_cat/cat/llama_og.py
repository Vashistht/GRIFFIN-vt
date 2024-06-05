# Adapted from Hugging Face implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from griffin.utils import select_neurons
import matplotlib.pyplot as plt

def get_llama_griffin(model,  k_schedule, threshold_list): 
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = LlamaMLP(config, k_schedule[i])
        t = threshold_list[i]
        new_mlp.gate_proj = l.mlp.gate_proj
        new_mlp.up_proj = l.mlp.up_proj
        new_mlp.down_proj = l.mlp.down_proj
        new_mlp.act_fn = l.mlp.act_fn
        new_mlp.threshold = t
        l.mlp = new_mlp
    
    return model


class LlamaMLP(nn.Module):
    def __init__(self, config, k_factor, threshold=0.15):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        self.threshold = threshold
        self.k_factor = k_factor

    # def prepare_reduced_weights(self, topk_indices):
    #     assert topk_indices.shape[0] == 1 # Batch size 1
        
    #     self.gate_proj_reduced = nn.Linear(self.gate_proj.weight.data.shape[1], len(topk_indices), bias=False)
    #     self.up_proj_reduced = nn.Linear(self.up_proj.weight.data.shape[1], len(topk_indices), bias=False)
    #     self.down_proj_reduced = nn.Linear(len(topk_indices), self.down_proj.weight.data.shape[0], bias=False)
    #     topk_indices = topk_indices[0]

    #     self.gate_proj_reduced.weight.data = self.gate_proj.weight.data[topk_indices]
    #     self.up_proj_reduced.weight.data = self.up_proj.weight.data[topk_indices]
    #     self.down_proj_reduced.weight.data = self.down_proj.weight.data[:, topk_indices]
    

    def forward(self, x):
            k_factor = self.k_factor
            if self.k_factor ==0:
                down_proj =self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                return down_proj
            if x.shape[1] > 1: # for the prompt
                v = self.act_fn(self.gate_proj(x))
                v_abs = torch.abs(v)
                Mask = (torch.zeros_like(v_abs) == 1) ## initialize a mask to be all False
                Mask = v_abs > self.threshold
                Mask = Mask.float()
                x1 = (v*Mask)* (self.up_proj(x) * Mask)
                y = self.down_proj(x1)


                self.up_proj(x)
        return down_proj

    
    def get_threshold(self, x, sparsity_level=0.7):
        v = self.act_fn(self.gate_proj(x))
        v = torch.abs(v)
        
        sorted_v, _ = torch.sort(v.flatten())
        # cdf = torch.arange(1, len(sorted_v) + 1) / len(sorted_v)
        
        # idx = torch.nonzero(cdf >= sparsity_level, as_tuple=True)[0]
        
        # if idx.numel() == 0:
        #     t = sorted_v[-1]
        # else:
        #     t = sorted_v[idx[0]]  
        return sorted_v
    
    def plot_threshold(self, x):
        t = self.get_threshold(x)
        print(t)
        v = self.act_fn(self.gate_proj(x))
        v = torch.abs(v)
        hist, bins = torch.histogram(v, bins=100)
        plt.figure(figsize=(8, 6))
        plt.plot(bins[:-1], hist, marker='o', linestyle='-', color='blue', label='Absolute Post-Activations')
        plt.xlabel('Absolute Post-Activations')
        plt.ylabel('Frequency')
        plt.title('Histogram of Absolute Post-Activations')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # def prune_cats(self, x):
    #         v = self.act_fn(self.gate_proj(x))
    #         v = torch.abs(v)
    #         t = 0.15 # for the 70% sparsity
    #         # GRIFFIN Expert Selection
    #         if self.config.selection_method != 'magnitude' and k_factor > 0.0: ###
    #             # print('expert selection', x.shape)
    #             k = int(int_states.shape[-1] * k_factor)
    #             neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1))).norm(dim=1) # B, D
    #             topk_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k)
    #             self.prepare_reduced_weights(topk_indices)
                
    #         down_proj = self.down_proj(int_states)
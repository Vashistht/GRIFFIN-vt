# Adapted from Hugging Face implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_llama_cats(model,  k_schedule,threshold_list):
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = LlamaMLP(config, k_schedule[i], threshold_list[i])
        new_mlp.gate_proj = l.mlp.gate_proj
        new_mlp.up_proj = l.mlp.up_proj
        new_mlp.down_proj = l.mlp.down_proj
        new_mlp.act_fn = l.mlp.act_fn
        new_mlp.threshold = threshold_list[i]
        l.mlp = new_mlp
    return model


class LlamaMLP(nn.Module):
    def __init__(self, config, k_factor, threshold):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        
        self.k_factor = k_factor
        self.threshold = threshold


    def forward(self, x, is_cats=True):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            k_factor = self.k_factor
            if (self.k_factor ==0) or (is_cats is False):
                down_proj =self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                return down_proj
            else:
                x = x [:, -1, :]
                v = self.act_fn(self.gate_proj(x))
                v_abs = torch.abs(v)
                Mask = (torch.zeros_like(v_abs) == 1) ## initialize a mask to be all False 1, 11008
                Mask = v_abs > self.threshold
                Mask = Mask.float()
                v_masked = v.mul(Mask)
                assert v_masked.shape == v.shape
                import pdb; pdb.set_trace()
                W_up_masked = (self.up_proj.weight.data).mul(Mask.T)# (11008x4096) *(1, 11008)
                # (W_up_masked*x) = [11008, 4096]
                W_down_masked = (self.down_proj.weight.data).mul(Mask.T)

                x1 = (W_up_masked.mul(x)).mul(v_masked.T) # [11008, 4096]
                y = (x1).mul(W_down_masked) # [11008, 4096]
                assert y.shape ==  self.up_proj.weight.data.shape
            return y
        
        print(f'W_down_proj: {self.down_proj.shape}')
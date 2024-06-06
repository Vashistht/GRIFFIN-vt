#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
import argparse
import logging


import numpy as np
from griffin.llama_og import get_llama_griffin
# from griffin.gemma import get_gemma_griffin
# from griffin.mistral import get_mistral_griffin
# from griffin.opt import get_opt_griffin

import torch
import torch.nn.functional as F
import json
import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from rouge import Rouge



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_distribution(logits, temperature, epsilon=1e-8):
    logits = logits  # Move logits to the device
    logits /= (temperature + epsilon)
    probability = F.softmax(logits, dim=-1)
    return probability

# %%
def sample(logits, temperature):
    output = get_distribution(logits, temperature)
    output = torch.multinomial(output, num_samples=1)
    return output.squeeze(1)



def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

models_sizes_dict = {
    'opt': ['125m', '350m', '1.3b', '2.7b', '6.7b', '13b', '30b', '66b'],
    'llama2': ['7b', '13b', '70b'],
    'relu_llama2': ['7B', '13B', '70B'],
    'gemma': ['2b', '7b'],
    'mistral': ['7B'],
}

hugging_name_dict = {
    'opt': lambda x: f'facebook/opt-{x}',
    'llama2': lambda x: f'meta-llama/Llama-2-{x}-hf', 
    'relu_llama2': lambda x: f"SparseLLM/ReluLLaMA-{x}",
    'gemma': lambda x: f'google/gemma-{x}',
    'mistral': lambda x: f'mistralai/Mistral-{x}-v0.1',
}


modify_dict = {
    # 'opt': get_opt_griffin,
    'llama2': get_llama_griffin,
    'relu_llama2': get_llama_griffin,
    # 'gemma': get_gemma_griffin,
    # 'mistral': get_mistral_griffin,
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


# greedy decoding by calling _greedy_search() if num_beams=1 and do_sample=False
# multinomial sampling by calling _sample() if num_beams=1 and do_sample=True

def main(dataset='cnn', shots=1, model_arch='llama2', model_size=0, cache_dir=None,
         density=0.5, selection_method='topk', sample_num=1, max_length=-1,
         k=0, max_tokens=128, seed=42, temp=0.3, greedy=False, device='cuda:1', forward= True): # @vashisthtiwari 
    
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(dataset=dataset, shots=shots, model_arch=model_arch, model_size=model_size,
                cache_dir=cache_dir, density=density, selection_method=selection_method,
                sample_num=sample_num, max_length=max_length, k=k, max_tokens=max_tokens,
                seed=seed, temp=temp, greedy=greedy,
                device=device, forward=forward)
    
    set_seed(args)


    model_size_name = models_sizes_dict[args.model_arch][args.model_size]
    
    config = AutoConfig.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(hugging_name_dict[args.model_arch](model_size_name))

    print("PARAMS: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    schedule_k = [args.density for _ in range(config.num_hidden_layers)]
    
    if args.density < 1: # never enters gen so try .999 or something
        model.config.mode = 'gen'
        model.config.selection_method = args.selection_method
        model = modify_dict[args.model_arch](model, schedule_k)
    
    model.half()
    model.eval().to(args.device)
    
    if args.dataset == 'cnn':
        # input_paths = [f'../data/cnn_data/cnn_dailymail_{args.shots}shot.jsonl']
        input_paths = [f'/home/vashistt/Desktop/GRIFFIN-vt/data/cnn_data/cnn_dailymail_{args.shots}shot.jsonl']
    elif args.dataset == 'xsum':
        # input_paths = [f'../data/xsum_data/xsum_{args.shots}shot.jsonl']
        input_paths = [f'/home/vashistt/Desktop/GRIFFIN-vt/xsum_data/xsum_{args.shots}shot.jsonl']
    else:
        raise NotImplementedError
    
    if args.max_length == -1:
        args.max_length = config.max_position_embeddings
        # args.max_length = 

    # Logging or return
    print(args)
    if args.max_length == -1:
        args.max_length = config.max_position_embeddings
    logger.info(args)
    
    requests = []
    for input_path in input_paths:
         with open(input_path, 'r') as f:
             for line in f:
                 if line.strip() != '':
                     requests.append(json.loads(line))

    requests = requests[:args.sample_num]


    skipped=0
    n_v_all_layer = []
    with torch.no_grad():
        for i, request in enumerate(tqdm.tqdm(requests)):        
            stop = ['###']
            temperature = args.temp
            # prompt = request['article']
            prompt = 'The United States of America is '
            print('prompt:', prompt)
            label = request['summary_gt']
            max_tokens = args.max_tokens
            result = {}
            
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            original_input_len = len(input_ids[0])

            if len(input_ids[0]) > args.max_length-max_tokens:
                skipped+=1
                print('skipped', skipped)

            else:
                print('using model forward')
                v_list_perlayer = []
                for i, l in enumerate(model.model.layers):
                    v = l.mlp.get_threshold(input_ids)
                    print(i, v.shape)
                    v_list_perlayer.append(v)
            n_v_all_layer.append(v_list_perlayer)
    

        n_v_all_layer = torch.tensor(n_v_all_layer) 
        n_v_all_layer_flat = n_v_all_layer.flatten()
        print(n_v_all_layer_flat.shape)
        # Flatten the n_v_all_layer tensor

        # Plot the histogram
        plt.figure(figsize=(8, 6))
        plt.hist(n_v_all_layer_flat.cpu().numpy(), bins=50, density=True, alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Histogram of n_v_all_layer')
        plt.grid(True)
        plt.show()

        # Calculate the CDF
        sorted_values = np.sort(n_v_all_layer_flat.cpu().numpy())
        cdf = np.cumsum(sorted_values) / len(sorted_values)

        # Plot the CDF
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_values, cdf)
        plt.xlabel('Value')
        plt.ylabel('CDF')
        plt.title('Cumulative Distribution Function')
        plt.grid(True)
        plt.show()

    # Calculate the CDF greater than some k value
    k_value = 0.5  # Example k value, adjust as needed
    cdf_greater_than_k = 1 - np.interp(k_value, sorted_values, cdf)
    print(f"CDF greater than {k_value}: {cdf_greater_than_k}")
        
                
                
    return model
    # return model, tokenizer, input_paths, args


# Now you can call this function directly with parameters from your notebook:
# model, tokenizer, input_paths, args = main(dataset="xsum", shots=10, model_arch="gpt2", model_size=1)

if __name__ == "__main__":
    main()
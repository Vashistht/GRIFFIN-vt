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
from griffin.llama import get_llama_griffin
from griffin.gemma import get_gemma_griffin
from griffin.mistral import get_mistral_griffin
from griffin.opt import get_opt_griffin

import torch
import torch.nn.functional as F
import json
import tqdm

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
    'opt': get_opt_griffin,
    'llama2': get_llama_griffin,
    'relu_llama2': get_llama_griffin,
    'gemma': get_gemma_griffin,
    'mistral': get_mistral_griffin,
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


# greedy decoding by calling _greedy_search() if num_beams=1 and do_sample=False
# multinomial sampling by calling _sample() if num_beams=1 and do_sample=True

def main(dataset='cnn', shots=1, model_arch='llama2', model_size=0, cache_dir=None,
         density=0.1, selection_method='topk', sample_num=1, max_length=-1,
         k=0, max_tokens=128, seed=42, temp=0.3, greedy=False, device='cuda:9', forward= False): # @vashisthtiwari 
    
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(dataset=dataset, shots=shots, model_arch=model_arch, model_size=model_size,
                cache_dir=cache_dir, density=density, selection_method=selection_method,
                sample_num=sample_num, max_length=max_length, k=k, max_tokens=max_tokens,
                seed=seed, temp=temp, greedy=greedy,
                device=device, forward=forward)
    
    set_seed(args)

    # if args.dataset == 'cnn':
       
    #     # input_paths = [f'../data/cnn_data/cnn_dailymail_{args.shots}shot.jsonl']
    #     input_paths = [f'/home/vashistt/Desktop/GRIFFIN-vt/data/cnn_data/cnn_dailymail_{args.shots}shot.jsonl']
    # elif args.dataset == 'xsum':
    #     # input_paths = [f'../data/xsum_data/xsum_{args.shots}shot.jsonl']
    #     input_paths = [f'/home/vashistt/Desktop/GRIFFIN-vt/xsum_data/xsum_{args.shots}shot.jsonl']
    # else:
    #     raise NotImplementedError

    model_size_name = models_sizes_dict[args.model_arch][args.model_size]
    
    config = AutoConfig.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(hugging_name_dict[args.model_arch](model_size_name), use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(hugging_name_dict[args.model_arch](model_size_name))

    print("PARAMS: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    schedule_k = [args.density for _ in range(config.num_hidden_layers)]
    
    if args.density < 1:
        model.config.mode = 'gen'
        model.config.selection_method = args.selection_method
        model = modify_dict[args.model_arch](model, schedule_k)
    
    model.half()
    model.eval().to(args.device)
    
    # if args.max_length == -1:
    #     args.max_length = config.max_position_embeddings
    #     # args.max_length = 

    # # Logging or return
    # print(args)
    # if args.max_length == -1:
    #     args.max_length = config.max_position_embeddings
    # logger.info(args)

    # requests = []
    # for input_path in input_paths:
    #      with open(input_path, 'r') as f:
    #          for line in f:
    #              if line.strip() != '':
    #                  requests.append(json.loads(line))

    # requests = requests[:args.sample_num]

    # results = []
    # rouge = Rouge()

    # seq_lens = []
    # rouge1_score_list = []
    # rouge2_score_list = []
    # rougel_score_list = []

    # skipped=0
    
    # with torch.no_grad():
    #     for i, request in enumerate(tqdm.tqdm(requests)):        
    #         stop = ['###']
    #         temperature = args.temp
    #         # prompt = request['article']
    #         prompt = 'The United States of America is '
    #         print('prompt:', prompt)
    #         label = request['summary_gt']
    #         max_tokens = args.max_tokens
    #         result = {}
    #         if args.model_arch == 'gemma':
    #             input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    #         else:
    #             input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
    #             original_input_len = len(input_ids[0])

    #         if len(input_ids[0]) > args.max_length-max_tokens:
    #             skipped+=1
    #             print('skipped', skipped)

    #         else:
    #             if args.forward == True:
    #                 print('using model forward')
    #                 total_len = input_ids.shape[1]
    #                 target_len = total_len + max_tokens
    #                 model(input_ids).logits[:, -1, :]
    #                 while total_len < target_len :
    #                     output_sequences = model(input_ids).logits[:, -1, :]
    #                     sample_token = sample(output_sequences, temperature=temp)
    #                     input_ids = torch.cat([input_ids, sample_token.unsqueeze(0)], dim=-1)
    #                     total_len += 1
    #                 generate_text = tokenizer.decode(input_ids[0][original_input_len:])
    #                 print('generated using forward:', generate_text)
    #             else:
    #                 print("using model.generate")
    #                 output_sequences = model.generate(
    #                     input_ids=input_ids,
    #                     max_length=max_tokens + len(input_ids[0]),
    #                     temperature=temperature,
    #                     top_k=args.k,
    #                     top_p=1,
    #                     do_sample=not args.greedy,
    #                     num_return_sequences=1,
    #                     return_dict_in_generate=True, output_scores=True,
    #                     use_cache= True
    #                     )

    #                 tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
    #                 logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
    #                 top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

    #                 generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
    #                 # generate_text = generate_text[: generate_text.find(stop[0])]
    #                 print('generated using .generate:', generate_text)
                    
    return model
    # return model, tokenizer, input_paths, args



# Now you can call this function directly with parameters from your notebook:
# model, tokenizer, input_paths, args = main(dataset="xsum", shots=10, model_arch="gpt2", model_size=1)

if __name__ == "__main__":
    main()
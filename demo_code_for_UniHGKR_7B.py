import argparse
import csv
import torch
import json
import os
import heapq
import pickle
import time
import logging
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
# from sentence_transformers import models
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, overload
from torch.amp import autocast
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



general_ins = "Given a question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst = "Given a question, retrieve relevant evidence that can answer the question from {} sources: "

general_ins_with_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from {} sources: "



def get_query_inputs(queries, tokenizer, device, max_length=256):
    prefix = general_ins
    #  or prefix = one of [single_source_inst, general_ins_with_domain ,single_source_inst_domain]
    
    suffix = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    queries_inputs = []
    for query in queries:
        inputs = tokenizer(query,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        queries_inputs.append(inputs)

    inputs = tokenizer.pad(
        queries_inputs,
        padding=True,
        max_length=max_length,
        pad_to_multiple_of=8,
        return_tensors='pt',
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    return inputs


def get_passage_inputs(passages, tokenizer, device, max_length=256):
    prefix = '"'
    suffix = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    passages_inputs = []
    for passage in passages:
        inputs = tokenizer(passage,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        passages_inputs.append(inputs)
    inputs = tokenizer.pad(
        passages_inputs,
        padding=True,
        max_length=max_length,
        pad_to_multiple_of=8,
        return_tensors='pt',
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    return inputs



def main():

    model_name = args.model_name

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    logging.info("Convert model to FP16")
    model.half()
    
    logging.info("Move model to gpu")
    model = model.to(device)

    logging.info(f"device:{device}")
    
    
    eaxmple_inputs = get_query_inputs(["who found the Wise Club ?"], tokenizer, device)
    eaxmple_outputs = model(**eaxmple_inputs, return_dict=True, output_hidden_states=True)
    example_embeddings = eaxmple_outputs.hidden_states[-1][:, -8:, :]
    example_embeddings = torch.mean(example_embeddings, dim=1)
    example_embeddings = torch.nn.functional.normalize(example_embeddings, dim=-1)
    

#    for passages(or evidences) in corpus, please use the function: get_passage_inputs
#    example_inputs is transformed into example_embeddings, which is the same as the above query.



if __name__ == "__main__":
    main()

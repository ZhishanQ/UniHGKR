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

source_map = {
    'kb':'Knowledge Graph triple',
    'info':'Infobox',
    'text':'Text',
    'table':'Table'
}

general_ins = "Given a question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst = "Given a question, retrieve relevant evidence that can answer the question from {} sources: "

general_ins_with_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from {} sources: "

question2domain_path = "../rag/finetune_with_ins_dataset/question2domain.json"
ins2diverse_ins_list_path = "../rag/finetune_with_ins_dataset/ins2diverse_ins_list.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Script to encode sentences and perform semantic search.")
    parser.add_argument('--gpus', type=str, default="0", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument('--model_name', type=str, required=True, help="Path to the sentence transformer model.")
    parser.add_argument('--convmix_test_set', type=str, required=True, choices=['train', 'dev', 'test', 'train_dev'], help="Specify the convmix test set to use.")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for encoding the corpus.")
    parser.add_argument('--save_embedding', type=int, required=True, help="Whether to save cache.")
    parser.add_argument('--debug_flag', type=int, default=0 )
    parser.add_argument('--use_domain_tips', type=int, required=True, choices=[0, 1])
    parser.add_argument('--use_diverse_ins', type=int, required=True, choices=[0, 1])
    parser.add_argument('--use_single_source', type=str, required=True, choices=['all', 'kb', 'info', 'text', 'table'])
    return parser.parse_args()



def get_query_inputs(queries, tokenizer, device, max_length=256):
    prefix = '"'
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


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    device_be_used,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 400000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
) -> List[List[Dict[str, Union[int, float]]]]:
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    Args:
        query_embeddings (Tensor): A 2 dimensional tensor with the query embeddings.
        corpus_embeddings (Tensor): A 2 dimensional tensor with the corpus embeddings.
        query_chunk_size (int, optional): Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory. Defaults to 100.
        corpus_chunk_size (int, optional): Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory. Defaults to 500000.
        top_k (int, optional): Retrieve top k matching entries. Defaults to 10.
        score_function (Callable[[Tensor, Tensor], Tensor], optional): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Dict[str, Union[int, float]]]]: A list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    logging.info(f"corpus_embeddings.device :{corpus_embeddings.device}")
    logging.info(f"query_embeddings.device :{query_embeddings.device}")

    logging.info(f"corpus_embeddings[0].dtype :{corpus_embeddings[0].dtype}")
    logging.info(f"query_embeddings[0].dtype :{query_embeddings[0].dtype}")
    

    if corpus_embeddings.device != query_embeddings.device:
        logging.info(f"move query_embeddings to corpus_embeddings.device")
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    # return 0

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in tqdm(range(0, len(query_embeddings), query_chunk_size)):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size].to(device_be_used),
                corpus_embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size].to(device_be_used),
            )

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {"corpus_id": corpus_id, "score": score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)

    return queries_result_list



def main():
    args = parse_args()
    logging.info(f"args: {args}")
    # Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    batch_size = args.batch_size

    with open(question2domain_path, "r") as fp:
        question2domain = json.load(fp)

    with open(ins2diverse_ins_list_path, "r") as fp:
        ins2diverse_ins_list = json.load(fp)

    model_name = args.model_name
    model_name_base = model_name.split('/')[-1]
    save_embedding = args.save_embedding

    convmix_test_set = args.convmix_test_set
    convmix_test_set_path = f"../rag/convmix_dataset/annotated_{convmix_test_set}.json"
    retrived_evi_and_que_path = f"../rag/retrival_results/{model_name_base}_retrived_evi_and_que_{convmix_test_set}_{args.use_single_source}.json"

    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    logging.info("Convert model to FP16")
    model.half()  # Convert model to FP16
    logging.info("Move model to gpu")
    model = model.to(device)

    logging.info(f"device:{device}")
    dataset_name = "compmix_full_set_all_fp16"
    dataset_path = '../rag/kb_wikipedia_mixed_rd.pickle'
    embedding_cache_path = f"../rag/embedding_cache/test-embeddings-{model_name_base}-size-{dataset_name}.pkl"
    # test-embeddings-LLARA-passage-size-compmix_full_set_all_fp16.pkl

    if args.debug_flag:
        eaxmple_inputs = get_passage_inputs(["who found the Wise Club ?"], tokenizer, device)
        eaxmple_outputs = model(**eaxmple_inputs, return_dict=True, output_hidden_states=True)
        example_embeddings = eaxmple_outputs.hidden_states[-1][:, -8:, :]
        example_embeddings = torch.mean(example_embeddings, dim=1)
        example_embeddings = torch.nn.functional.normalize(example_embeddings, dim=-1)
        print(type(example_embeddings))
        print(type(example_embeddings[0]))
        print(example_embeddings.dtype)
        print(example_embeddings[0].dtype)
        example_embeddings = example_embeddings.cpu()
        print(example_embeddings[0].dtype)
        logging.info(f"dimension of embeddings: {len(example_embeddings[0])}")

    # exit(0)

    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):

        logging.info(f"Please embed corpus first for llm embedder.")

    else:
        logging.info(f"Load pre-computed embeddings from disc: {embedding_cache_path}")

        with open(embedding_cache_path, "rb") as fIn:
            corpus_embeddings = pickle.load(fIn)

        # Get all unique sentences from the file
        corpus_sentences = list()
        with open(dataset_path, "rb") as fIn:
            data = pickle.load(fIn)
            for item in data:
                corpus_sentences.append(item["evidence_text"])

        if len(corpus_sentences) != len(corpus_embeddings):
            logging.warning(f"Length mismatch: corpus_sentences has length {len(corpus_sentences)}, but corpus_embeddings has length {len(corpus_embeddings)}")
            if not args.debug_flag:
               exit(0)

        logging.info(f"corpus_embeddings.device :{corpus_embeddings.device}")
        
        
    logging.info(type(corpus_embeddings))
    logging.info(type(corpus_embeddings[0]))
    logging.info(len(corpus_embeddings))
    logging.info(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")

    with open(convmix_test_set_path, "r") as fp:
        convmix_test_set_data = json.load(fp)

    top_k_hits = 100
    logging.info(f"top_k_hits: {top_k_hits}")
    logging.info(f"retrived_evi_and_que_path: {retrived_evi_and_que_path}")
    logging.info(f"convmix_test_set_path: {convmix_test_set_path}")
    
    retrived_evi_and_que = []

    questions = []
    question_info = []
    
    logging.info("loading queries...")
    for item in tqdm(convmix_test_set_data):
        for que in item['questions']:
            inp_question = que['question']
            if 'completed' in que:
                inp_question = que['completed']

            if args.use_domain_tips == 0:
                if args.use_single_source == 'all':
                    query_tmp = general_ins + inp_question
                else:
                    query_tmp = single_source_inst.format(source_map[args.use_single_source]) + inp_question
            else:
                if args.use_single_source == 'all':
                    static_ins = general_ins_with_domain.format(question2domain[inp_question])
                else:
                    static_ins = single_source_inst_domain.format(question2domain[inp_question], source_map[args.use_single_source])

                if args.use_diverse_ins == 0:
                    query_tmp = static_ins + inp_question
                else:
                    assert static_ins.endswith("sources: ")
                    query_tmp = random.choice(ins2diverse_ins_list[static_ins.rstrip()]) + " " + inp_question

            questions.append(query_tmp)
            question_info.append(que)
    

    logging.info("encoding queries...")
    all_question_embeddings = []
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_sentences = questions[i:i + batch_size]
        batch_inputs = get_query_inputs(batch_sentences, tokenizer, device)

        with torch.no_grad():
            with autocast('cuda'):  # Enable mixed precision
                question_outputs = model(**batch_inputs, return_dict=True, output_hidden_states=True)
                question_embeddings = question_outputs.hidden_states[-1][:, -8:, :]
                question_embeddings = torch.mean(question_embeddings, dim=1)
                question_embeddings = torch.nn.functional.normalize(question_embeddings, dim=-1)
                all_question_embeddings.append(question_embeddings.cpu())
    
    all_question_embeddings = torch.cat(all_question_embeddings, dim=0)
    logging.info(f"all_question_embeddings.device :{all_question_embeddings.device}")


    logging.info("semantic_search queries...")
    all_hits = semantic_search(all_question_embeddings, corpus_embeddings, device_be_used=device, top_k=top_k_hits, score_function=util.dot_score)

    for idx, hits in enumerate(all_hits):
        correct_hits_ids = set([hit["corpus_id"] for hit in hits])
        top_evidences = [data[corpus_id] for corpus_id in correct_hits_ids]

        retrived_evi_and_que.append({
            "correct_hits_ids": list(correct_hits_ids),  # Convert set to list here
            "question": question_info[idx],
            "top_evidences": top_evidences
        })

    logging.info("writing retrived_evi_and_que ...")
    with open(retrived_evi_and_que_path, 'w') as file:
        json.dump(retrived_evi_and_que, file, indent=4, ensure_ascii=False)
    
    logging.info("done.")



if __name__ == "__main__":
    main()

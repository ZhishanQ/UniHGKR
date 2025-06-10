import argparse
import csv
import random

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
from sentence_transformers import models
from torch import Tensor
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, overload

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

source_map = {
    'kb':'Knowledge Graph triple',
    'info':'Infobox',
    'text':'Text',
    'table':'Table'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Script to encode sentences and perform semantic search.")
    parser.add_argument('--gpus', type=str, default="0", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument('--model_name', type=str, required=True, help="Path to the sentence transformer model.")
    parser.add_argument('--convmix_test_set', type=str, required=True, choices=['train', 'dev', 'test'],
                        help="Specify the convmix test set to use (train, dev, test).")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for encoding the corpus.")
    parser.add_argument('--save_embedding', type=int, required=True, choices=[0, 1], help="Whether to save cache.")
    parser.add_argument('--corpus_id', type=int, default=0,
                        help="Choose which corpus be used")
    parser.add_argument('--use_domain_tips', type=int, required=True, choices=[0, 1])
    parser.add_argument('--use_diverse_ins', type=int, required=True, choices=[0, 1])
    parser.add_argument('--debug_flag', type=int, required=True, choices=[0, 1])
    parser.add_argument('--use_single_source', type=str, required=True, choices=['all', 'kb', 'info', 'text', 'table'])

    return parser.parse_args()


# You can change query_chunk_size and corpus_chunk_size to increase the speed, but it requires more memory. 
# Also, you should decrease query_chunk_size and corpus_chunk_size if you run out of memory.
def semantic_search(
        query_embeddings: Tensor,
        corpus_embeddings: Tensor,
        query_chunk_size: int = 100, 
        corpus_chunk_size: int = 2000000,
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
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in tqdm(range(0, len(query_embeddings), query_chunk_size)):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx: query_start_idx + query_chunk_size].to("cuda"),
                corpus_embeddings[corpus_start_idx: corpus_start_idx + corpus_chunk_size].to("cuda"),
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


general_ins = "Given a question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst = "Given a question, retrieve relevant evidence that can answer the question from {} sources: "

general_ins_with_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from {} sources: "

question2domain_path = "meta_data/question2domain.json"
ins2diverse_ins_list_path = "meta_data/ins2diverse_ins_list.json"


def eval_fun(model, corpus_embeddings, corpus_sentences, data, model_name_base, convmix_test_set_path,
             retrived_evi_and_que_path, args):
    with open(convmix_test_set_path, "r") as fp:
        convmix_test_set_data = json.load(fp)

    with open(question2domain_path, "r") as fp:
        question2domain = json.load(fp)

    with open(ins2diverse_ins_list_path, "r") as fp:
        ins2diverse_ins_list = json.load(fp)

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

            if model_name_base.startswith("hkunlp--instructor"):
                query_tmp = "Represent the Wikipedia question for retrieving supporting documents:" + inp_question
            
            if model_name_base.endswith("bge-base-en-v1.5") or model_name_base.endswith("bge-base-en"):
                query_tmp = "Represent this sentence for searching relevant passages:" + inp_question
            
            questions.append(query_tmp)
            question_info.append(que)

    logging.info(f"questions top 10:{questions[:10]}")
    logging.info("encoding queries...")
    question_embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    logging.info("semantic_search queries...")
    all_hits = semantic_search(question_embeddings, corpus_embeddings, top_k=top_k_hits, score_function=util.dot_score)

    # for idx, hits in enumerate(all_hits):
    #     correct_hits_ids = set([hit["corpus_id"] for hit in hits])
    #     top_evidences = [data[corpus_id] for corpus_id in correct_hits_ids]

    #     retrived_evi_and_que.append({
    #         "correct_hits_ids": list(correct_hits_ids),  # Convert set to list here
    #         "question": question_info[idx],
    #         "top_evidences": top_evidences
    #     })
    for idx, hits in enumerate(all_hits):
        # top_evidences = []
        # for hit in hits:
        #     evidence = data[hit["corpus_id"]]
        #     evidence["score"] = hit["score"]  # Add the score to the evidence dictionary
        #     top_evidences.append(evidence)

        # correct_hits = [{"corpus_id": hit["corpus_id"], "score": hit["score"]} for hit in hits]
        # top_evidences = [{"evidence": data[hit["corpus_id"]], "score": hit["score"]} for hit in correct_hits]

        correct_hits_ids = set([hit["corpus_id"] for hit in hits])
        top_evidences = [data[corpus_id] for corpus_id in correct_hits_ids]

        retrived_evi_and_que.append({
            "correct_hits_ids": list(correct_hits_ids),  # Convert set to list here
            "question": question_info[idx],
            "top_evidences": top_evidences
        })

        # retrived_evi_and_que.append({
        #     "correct_hits_ids": [hit["corpus_id"] for hit in correct_hits],
        #     "question": question_info[idx],
        #     "top_evidences": top_evidences
        # })

    logging.info("writing retrived_evi_and_que ...")
    with open(retrived_evi_and_que_path, 'w') as file:
        json.dump(retrived_evi_and_que, file, indent=4, ensure_ascii=False)

    logging.info("done.")


def main():
    args = parse_args()
    logging.info(f"args: {args}")

    # Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    my_batch_size = args.batch_size

    model_name = args.model_name
    model_name_base = model_name.split('/')[-1]
    save_embedding = args.save_embedding

    convmix_test_set = args.convmix_test_set
    convmix_test_set_path = f"../CompMix_IR/ConvMix_annotated/annotated_{convmix_test_set}.json"
    

    # Check if the directory "1_Pooling" exists in the model_name directory
    if os.path.isdir(os.path.join(model_name, "1_Pooling")):
        # If "1_Pooling" exists, load the model this way
        model = SentenceTransformer(model_name)
    else:
        # If "1_Pooling" does not exist, load the model this way
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if args.corpus_id == 0:
        dataset_name = "compmix_full_set"
        dataset_path = './kb_wikipedia_mixed_rd.pickle' # change this to the path of CompMix-IR corpus
    elif args.corpus_id == 1:
        dataset_name = "compmix_full_set_with_ins"
        dataset_path = './kb_wikipedia_mixed_rd_with_instructions.pickle'
        logging.warning('using kb_wikipedia_mixed_rd_with_instructions')
    else:
        dataset_name = "compmix_full_set_new_{}".format(args.corpus_id)
        dataset_path = './kb_wikipedia_mixed_rd_new_{}.pkl'.format(args.corpus_id)

    retrived_evi_and_que_path = f"retrival_results/{model_name_base}_retrived_evi_and_que_{convmix_test_set}_{dataset_name}_{args.use_single_source}.json"
    embedding_cache_path = f"embedding_cache/test-embeddings-{model_name_base}-size-{dataset_name}.pkl"

    embedding_sample = model.encode("who found the Wise Club ?", normalize_embeddings=True)
    embedding_size = len(embedding_sample)  # Size of embeddings
    logging.info("max_seq_length:{}".format(model.get_max_seq_length()))
    logging.info(f"embedding_size: {embedding_size}")

    if model_name_base.startswith("hkunlp--instructor"):
        logging.info("we are using instructor' ins templates ! ")

    if model_name_base.endswith("bge-base-en-v1.5") or model_name_base.endswith("bge-base-en"):
         logging.info("we are using bge' ins templates ! ")

    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):

        logging.info(f"load evidence text from : {dataset_path}")

        corpus_sentences = list()
        with open(dataset_path, "rb") as fIn:
            data = pickle.load(fIn)
            for item in data:
                if model_name_base.startswith("hkunlp--instructor"):
                    corpus_sentences.append("Represent the  Wikipedia document for retrieval:" + item["evidence_text"])
                else:
                    corpus_sentences.append(item["evidence_text"])

        logging.info(f"corpus_sentences len: {len(corpus_sentences)}")
        logging.info(f"my_batch_size:{my_batch_size}")
        logging.info("Encode the corpus. This might take a while")

        # Set the TOKENIZERS_PARALLELISM environment variable to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Start the multi-process pool on all available CUDA devices
        pool = model.start_multi_process_pool()

        if args.debug_flag:
            corpus_sentences = corpus_sentences[:100]

        # Compute the embeddings using the multi-process pool
        corpus_embeddings = model.encode_multi_process(corpus_sentences, pool, batch_size=my_batch_size,
                                                       normalize_embeddings=True)
        
        

        # Optional: Stop the processes in the pool
        model.stop_multi_process_pool(pool)

        # Convert embeddings to numpy
        corpus_embeddings = np.array(corpus_embeddings)
        if save_embedding > 0:
            logging.info("Store file on disc")
            with open(embedding_cache_path, "wb") as fOut:
                pickle.dump(corpus_embeddings, fOut)

    else:
        logging.info("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            corpus_embeddings = pickle.load(fIn)

        # Get all unique sentences from the file
        corpus_sentences = list()
        with open(dataset_path, "rb") as fIn:
            data = pickle.load(fIn)
            for item in data:
                corpus_sentences.append(item["evidence_text"])

        if len(corpus_sentences) != len(corpus_embeddings):
            logging.warning(
                f"Length mismatch: corpus_sentences has length {len(corpus_sentences)}, but corpus_embeddings has length {len(corpus_embeddings)}")

    logging.info(type(corpus_embeddings))
    logging.info(type(corpus_embeddings[0]))
    logging.info(len(corpus_embeddings))
    logging.info(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")

    # Call the evaluation function
    eval_fun(model, corpus_embeddings, corpus_sentences, data, model_name_base, convmix_test_set_path,
             retrived_evi_and_que_path, args)


if __name__ == "__main__":
    main()

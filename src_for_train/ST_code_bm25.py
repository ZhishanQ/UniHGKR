import argparse
import csv
import json
import os
import pickle
import time
import logging
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, overload

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Script to encode sentences and perform semantic search.")
    parser.add_argument('--gpus', type=str, default="0", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument('--convmix_test_set', type=str, required=True, choices=['train', 'dev', 'test'], help="Specify the convmix test set to use (train, dev, test).")
    parser.add_argument('--part', type=int, required=True, help="Specify which part of the questions to process (0-4).")

    return parser.parse_args()

# because the bm25 model with cpu is too slow, we split the questions into 5 parts and process them separately

def convert_to_native(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    return obj


def eval_fun(bm25, data, convmix_test_set_path, retrived_evi_and_que_path, part, total_parts):

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
            questions.append(inp_question)
            question_info.append(que)

    # Split questions into parts
    total_questions = len(questions)
    part_size = total_questions // total_parts
    start_idx = part * part_size
    end_idx = total_questions if part == total_parts - 1 else (part + 1) * part_size

    questions_part = questions[start_idx:end_idx]
    question_info_part = question_info[start_idx:end_idx]

    logging.info(f"Processing part {part + 1}/{total_parts} with {len(questions_part)} questions")

    logging.info("semantic_search queries...")

    for idx, inp_question in enumerate(tqdm(questions_part)):
        tokenized_query = inp_question.lower().split(' ')
        scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:top_k_hits]

        correct_hits_ids = set(top_k_indices)
        top_evidences = [data[corpus_id] for corpus_id in correct_hits_ids]

        retrived_evi_and_que.append({
            "correct_hits_ids": list(correct_hits_ids),  # Convert set to list here
            "question": question_info_part[idx],
            "top_evidences": top_evidences
        })

    retrived_evi_and_que = convert_to_native(retrived_evi_and_que)

    logging.info("writing retrived_evi_and_que ...")
    with open(retrived_evi_and_que_path, 'w') as file:
        json.dump(retrived_evi_and_que, file, indent=4, ensure_ascii=False)

    logging.info("done.")


def main():
    args = parse_args()
    logging.info(f"args: {args}")
    # Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    convmix_test_set = args.convmix_test_set
    part = args.part
    total_parts = 5
    convmix_test_set_path = f"convmix_dataset/annotated_{convmix_test_set}.json"
    retrived_evi_and_que_path = f"retrival_results/bm25_retrived_evi_and_que_{convmix_test_set}_part{part}.json"

    dataset_path = './kb_wikipedia_mixed_rd.pickle' # change this to the path of CompMix-IR corpus

    logging.info(f"load evidence text from : {dataset_path}")

    corpus_sentences = list()
    with open(dataset_path, "rb") as fIn:
        data = pickle.load(fIn)
        for item in data:
            corpus_sentences.append(item["evidence_text"])

    logging.info(f"corpus_sentences len: {len(corpus_sentences)}")

    # Tokenize the corpus
    tokenized_corpus = [doc.lower().split(' ') for doc in tqdm(corpus_sentences, desc="Tokenizing corpus")]

    # Initialize BM25
    logging.info(f"Initialize BM25")
    bm25 = BM25Okapi(tokenized_corpus)

    logging.info(f"Corpus loaded with {len(corpus_sentences)} sentences")

    # Call the evaluation function
    eval_fun(bm25, data, convmix_test_set_path, retrived_evi_and_que_path, part, total_parts)

if __name__ == "__main__":
    main()

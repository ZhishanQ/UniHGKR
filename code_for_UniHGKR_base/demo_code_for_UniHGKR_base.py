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
from sentence_transformers import models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



general_ins = "Given a question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst = "Given a question, retrieve relevant evidence that can answer the question from {} sources: "

general_ins_with_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from {} sources: "



def main():

    # Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model_name = "ZhishanQ/UniHGKR-base"
    model_name_base = model_name.split('/')[-1]

    model = SentenceTransformer(model_name)

    my_batch_size = 32
    embedding_sample = model.encode("who found the Wise Club ?", normalize_embeddings=True)
    embedding_size = len(embedding_sample)  # Size of embeddings
    logging.info(f"embedding_size: {embedding_size}") # 786

    # 1. embedding for queries

    questions = ["question 1", "question 2", "question 3", "your question list"]

    logging.info("preprocess queries...")

    # Prepend each question with the instruction
    updated_questions = [f"{general_ins}{question}" for question in questions]

    logging.info("encoding queries...")
    question_embeddings = model.encode(updated_questions, normalize_embeddings=True, show_progress_bar=True)

    # 2. embedding for corpus

    # Check if embedding cache path exists
    embedding_cache_path = "./your_embedding_cache_path"

    if not os.path.exists(embedding_cache_path):

        corpus_sentences = ["evidence 1", "evidence 2", "evidence 3", "your evidence corpus"]

        logging.info(f"corpus_sentences len: {len(corpus_sentences)}")
        logging.info(f"my_batch_size:{my_batch_size}")
        logging.info("Encode the corpus. This might take a while")

        # Set the TOKENIZERS_PARALLELISM environment variable to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Start the multi-process pool on all available CUDA devices
        # use multi GPU
        pool = model.start_multi_process_pool()

        # Compute the embeddings using the multi-process pool
        corpus_embeddings = model.encode_multi_process(corpus_sentences, pool, batch_size=my_batch_size,
                                                       normalize_embeddings=True)

        # Optional: Stop the processes in the pool
        model.stop_multi_process_pool(pool)

        # Convert embeddings to numpy
        corpus_embeddings = np.array(corpus_embeddings)

        logging.info("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump(corpus_embeddings, fOut)

    else:
        logging.info("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            corpus_embeddings = pickle.load(fIn)

    similarities = model.similarity(question_embeddings, corpus_embeddings)
    print(similarities.shape)


if __name__ == "__main__":
    main()

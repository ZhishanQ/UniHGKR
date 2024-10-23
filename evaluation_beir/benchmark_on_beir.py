from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import os, json

# Set the environment variable "CUDA_VISIBLE_DEVICES" to "0"
# This specifies which GPU devices to make visible to the application.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

general_ins = "Given a question, retrieve relevant evidence that can answer the question from all knowledge sources:"
single_source_inst = "Given a question, retrieve relevant evidence that can answer the question from Text sources:"


def evaluate_retrieval_model(dataset, retriever, ins):
    # Download and unzip dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = "./datasets"
    data_path = util.download_and_unzip(url, out_dir)

    # Load data
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # Modify queries with instructions
    queries_unihkgr = {key: ins + value for key, value in queries.items()}

    # Retrieve
    results = retriever.retrieve(corpus, queries_unihkgr)

    # Evaluate model
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    return ndcg, _map, recall, precision


def evaluate_retrieval_model_cqadupstack(dataset, retriever, ins):
    # Download and unzip dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = "./datasets"
    data_path = util.download_and_unzip(url, out_dir)

    # List of subfolders/domains in CQADupStack
    subfolders = [
        "android", "english", "gaming", "gis", "mathematica",
        "physics", "programmers", "stats", "tex", "unix",
        "webmasters", "wordpress"
    ]
    assert len(subfolders) == 12
    # Initialize accumulators for metrics
    ndcg_accum = {}
    map_accum = {}
    recall_accum = {}
    precision_accum = {}

    num_subfolders = len(subfolders)

    for subfolder in subfolders:
        logging.info(f"\nsubfolder: {subfolder}\n")
        # Load data for each subfolder
        subfolder_path = os.path.join(data_path, subfolder)
        corpus, queries, qrels = GenericDataLoader(data_folder=subfolder_path).load(split="test")

        # Modify queries with instructions
        queries_unihkgr = {key: ins + value for key, value in queries.items()}

        # Retrieve results
        results = retriever.retrieve(corpus, queries_unihkgr)

        # Evaluate metrics
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        # Accumulate NDCG metrics
        for k in ndcg.keys():
            ndcg_accum[k] = ndcg_accum.get(k, 0) + ndcg[k]

        # Accumulate MAP metrics
        for k in _map.keys():
            map_accum[k] = map_accum.get(k, 0) + _map[k]

        # Accumulate Recall metrics
        for k in recall.keys():
            recall_accum[k] = recall_accum.get(k, 0) + recall[k]

        # Accumulate Precision metrics
        for k in precision.keys():
            precision_accum[k] = precision_accum.get(k, 0) + precision[k]

    # Calculate average metrics
    ndcg_avg = {k: v / num_subfolders for k, v in ndcg_accum.items()}
    map_avg = {k: v / num_subfolders for k, v in map_accum.items()}
    recall_avg = {k: v / num_subfolders for k, v in recall_accum.items()}
    precision_avg = {k: v / num_subfolders for k, v in precision_accum.items()}

    # Return average metrics
    return ndcg_avg, map_avg, recall_avg, precision_avg


# model_path can be a local model or a HuggingFace model

model_path = "ZhishanQ/UniHGKR-base-beir"

model_base = model_path.split('/')[-1]
model = DRES(models.SentenceBERT(model_path), batch_size=128)

retriever = EvaluateRetrieval(model, score_function="cos_sim")

dataset_names =  ["scifact", "cqadupstack", "arguana", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa",
                  "nfcorpus","nq", "quora", "scidocs", "trec-covid", "webis-touche2020"]


NDCG10_result = {'model_name':model_base}
result_path = './result_path'

if not os.path.exists(result_path):
    os.makedirs(result_path)

for dataset_name in dataset_names[:1]:

    logging.info(f"dataset_name: {dataset_name}")

    if dataset_name == "cqadupstack":
        ndcg, _map, recall, precision = evaluate_retrieval_model_cqadupstack(dataset_name, retriever, single_source_inst)
    else:
        ndcg, _map, recall, precision = evaluate_retrieval_model(dataset_name, retriever, single_source_inst)

    results = {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision
    }
    file_name = f"{result_path}/{dataset_name}_{model_base}.json"

    with open(file_name, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    NDCG10_result[dataset_name+"_NDCG@10"] = ndcg['NDCG@10']

    with open(f"{result_path}/result_{model_base}.json", 'w') as json_file:
        json.dump(NDCG10_result, json_file, indent=4)

    # break

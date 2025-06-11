import argparse
import os
import pickle
import logging
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def parse_args():
    parser = argparse.ArgumentParser(description="Script to encode sentences and perform semantic search.")
    parser.add_argument('--gpus', type=str, default="0", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument('--model_name', type=str, required=True, help="Path to the sentence transformer model.")
    parser.add_argument('--convmix_test_set', type=str, required=True, choices=['train', 'dev', 'test'],
                        help="Specify the convmix test set to use (train, dev, test).")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for encoding the corpus.")
    parser.add_argument('--split_index', type=int, required=True, help="Index of the data split to process.")

    return parser.parse_args()




def main():
    args = parse_args()
    logging.info(args)
    
    # Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU
    gpu_id = args.gpus
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    batch_size = args.batch_size
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    model_name = args.model_name
    model_name_base = model_name.split('/')[-1]

    convmix_test_set = args.convmix_test_set
    # convmix_test_set_path = f"../convmix_dataset/annotated_{convmix_test_set}.json"
    # retrived_evi_and_que_path = f"../retrival_results/{model_name_base}_retrived_evi_and_que_{convmix_test_set}.json"

    # convmix_test_set_path = f"../convmix_dataset/annotated_{convmix_test_set}.json"
    # retrived_evi_and_que_path = f"../retrival_results/{model_name_base}_retrived_evi_and_que_{convmix_test_set}.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    model = model.to(device)

    dataset_name = "compmix_full_set"
    split_index = args.split_index
    dataset_path = f'../compmix_corpus_8_splits/kb_wikipedia_mixed_rd_split_{split_index}.pickle'
    embedding_cache_path = f"../embedding_cache/test-embeddings-{model_name_base}-size-{dataset_name}_split_{split_index}_fp16.pkl"

    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):

        logging.info(f"load evidence text from : {dataset_path}")

        with open(dataset_path, "rb") as fIn:
            corpus_sentences = pickle.load(fIn)

        logging.info(f"corpus_sentences len: {len(corpus_sentences)}")
        logging.info(f"my_batch_size:{batch_size}")
        logging.info("Encode the corpus. This might take a while")

        corpus_embeddings = []
        progress_path = f"../embedding_cache/progress_{model_name_base}_size_{dataset_name}_split_{split_index}.pkl"
        if os.path.exists(progress_path):
            logging.info("load progress...")
            with open(progress_path, "rb") as fIn:
                progress_data = pickle.load(fIn)
                corpus_embeddings = progress_data['corpus_embeddings']
                start_index = progress_data['start_index']
        else:
            start_index = 0

        cnt = 0 

        # exit(0)
        if len(corpus_embeddings) >0 :
            corpus_embeddings = [embedding.cpu() for embedding in corpus_embeddings]
            print(type(corpus_embeddings))
            print(type(corpus_embeddings[0]))

        for i in tqdm(range(start_index, len(corpus_sentences), batch_size)):
            batch_sentences = corpus_sentences[i:i + batch_size]
            batch_inputs = get_passage_inputs(batch_sentences, tokenizer, device)

            # Save progress every 1/20 of the total length
            # if cnt != 0 and cnt % (len(corpus_sentences) // (20*batch_size) ) == 0:
            #     logging.info(f"(len(corpus_sentences) // (20*batch_size) ): {(len(corpus_sentences) // (20*batch_size) )} , cnt: {cnt}, i:{i}")
            #     progress_data = {'corpus_embeddings': corpus_embeddings, 'start_index': i}
            #     with open(progress_path, "wb") as fOut:
            #         pickle.dump(progress_data, fOut)
            
            with torch.no_grad():
                with autocast('cuda'):  # Enable mixed precision
                    # compute batch embedding
                    batch_outputs = model(**batch_inputs, return_dict=True, output_hidden_states=True)
                    batch_embeddings = batch_outputs.hidden_states[-1][:, -8:, :]
                    batch_embeddings = torch.mean(batch_embeddings, dim=1)
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
                    corpus_embeddings.append(batch_embeddings.cpu())  # Move embeddings to CPU
                    # corpus_embeddings.append(batch_embeddings)

            cnt += 1
            

        # logging.info("convert to cpu")
        # corpus_embeddings = [embedding.cpu() for embedding in corpus_embeddings]
        corpus_embeddings = np.concatenate(corpus_embeddings, axis=0)

        logging.info("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump(corpus_embeddings, fOut)

    else:
        logging.info("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            corpus_embeddings = pickle.load(fIn)

    logging.info(type(corpus_embeddings))
    logging.info(type(corpus_embeddings[0]))
    logging.info(len(corpus_embeddings))
    logging.info(len(corpus_embeddings[0]))
    
    logging.info(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")
    logging.info("done")
    

if __name__ == "__main__":
    main()

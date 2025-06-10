import json
import pickle
import argparse
from tqdm import tqdm
from convinse_eval_part.evaluation import answer_presence, mrr_score, hit_at_k, hit_at_100
# from convinse.evaluation import answer_presence


# how to use?
# use this file to convert the retrieved evidence and question pairs into the format required for convmix evaluation.
# python src_for_train/convert_for_ers_format.py --model_name_base "UniHGKR-base" --convmix_test_set "test"

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--model_name_base', type=str, help='Base name of the model')
parser.add_argument('--convmix_test_set', type=str, default="test", choices=['train', 'dev', 'test','test_all','test__all'], help="Specify the convmix test set to use (train, dev, test).")
# parser.add_argument('--is_ers_format', type=int, required=True)

# Parse command line arguments
args = parser.parse_args()

# Use command line arguments
model_name_base = args.model_name_base
convmix_test_set = args.convmix_test_set
# is_ers_format = args.is_ers_format

print(f"model_name_base: {model_name_base}")
print(f"convmix_test_set: {convmix_test_set}")

add_suffix = ""
retrieved_evi_and_que_path = f"retrival_results/{model_name_base}_retrived_evi_and_que_{convmix_test_set}{add_suffix}.json"
print(f"retrieved_evi_and_que_path: {retrieved_evi_and_que_path}")

with open(retrieved_evi_and_que_path, "r") as fp:
    retrieved_evi_and_que = json.load(fp)

convinse_path = f"UniHGKR_training_data/uniHGKR_base_for_convinse_test_eval.json"
print(f"convinse style eval file path: {convinse_path}")

with open(convinse_path, "r") as fp:
    convinse_data = json.load(fp)

question_id_to_evi = {}

for item in tqdm(retrieved_evi_and_que):
    top_evidences = item['top_evidences']
    question_id = item['question']['question_id']
    question_id_to_evi[question_id] = top_evidences

cnt = 0
for conv in tqdm(convinse_data):
    for question in conv['questions']:
        # if len(question['top_evidences']) != 100:
        #     print(len(question['top_evidences']))
        # else:
        #     cnt += 1
        question['top_evidences'] = question_id_to_evi[question['question_id']]
        # assert len(question['top_evidences']) == 500
# print(cnt)

print("writing file....")

#  make sure the output path exists
import os
if not os.path.exists("dpr_info_with_ers_format"):
    os.makedirs("dpr_info_with_ers_format")

output_path = f"dpr_info_with_ers_format/{model_name_base}_eval_metrics_test{add_suffix}.json"
print(f"output_path: {output_path}")
with open(output_path, "w") as fp:
    json.dump(convinse_data, fp, indent=4)

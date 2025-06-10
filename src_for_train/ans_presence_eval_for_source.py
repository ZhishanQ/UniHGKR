import json
import pickle,os
import argparse
from tqdm import tqdm
from convinse_eval_part.evaluation import answer_presence,mrr_score,hit_at_k, hit_at_100

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--model_name_base', type=str, required=True, help='Base name of the model')
parser.add_argument('--convmix_test_set', type=str, required=True, choices=['train', 'dev', 'test','train_top_1k'], help="Specify the convmix test set to use (train, dev, test).")
parser.add_argument('--use_single_source', type=str, required=True, choices=['all', 'kb', 'info', 'text', 'table'])
parser.add_argument('--corpus_id',type=int, required=True )

# Parse command-line arguments
args = parser.parse_args()

# Use command-line arguments
model_name_base = args.model_name_base
convmix_test_set = args.convmix_test_set
use_single_source = args.use_single_source

print(f"model_name_base: {model_name_base}")
print(f"convmix_test_set: {convmix_test_set}")
print(f"use_single_source: {use_single_source}")

if args.corpus_id == 0:
    # dataset_name = ""
    dataset_name = "_compmix_full_set"
elif args.corpus_id == 1:
    # dataset_name = "compmix_full_set_with_ins"
    assert args.corpus_id != 1
else:
    dataset_name = "_compmix_full_set_new_{}".format(args.corpus_id)

retrieved_evi_and_que_path = f"retrival_results/{model_name_base}_retrived_evi_and_que_{convmix_test_set}{dataset_name}_{use_single_source}.json"
print(f"retrieved_evi_and_que_path: {retrieved_evi_and_que_path}")

# Try to read the file
if not os.path.exists(retrieved_evi_and_que_path):
    # Replace "_compmix_full_set" in the path with ""
    retrieved_evi_and_que_path2 = retrieved_evi_and_que_path.replace("_compmix_full_set", "")
    retrieved_evi_and_que_path3 = retrieved_evi_and_que_path.replace("_compmix_full_set", "_")
    
    print(f"File not found, trying alternative path: {retrieved_evi_and_que_path2}, {retrieved_evi_and_que_path3}")

    if os.path.exists(retrieved_evi_and_que_path2):
        retrieved_evi_and_que_path = retrieved_evi_and_que_path2
    elif os.path.exists(retrieved_evi_and_que_path3):
        retrieved_evi_and_que_path = retrieved_evi_and_que_path3
    else:
        raise FileNotFoundError(f"File not found: {retrieved_evi_and_que_path}")

with open(retrieved_evi_and_que_path, "r") as fp:
    retrieved_evi_and_que = json.load(fp)

with open(retrieved_evi_and_que_path, "r") as fp:
    retrieved_evi_and_que = json.load(fp)

with open('question_ids_in_compmix.json', "r") as fp:
    question_ids_in_compmix = json.load(fp)

# Define data source combinations
combinations = [
    "kb",
    "text",
    "table",
    "info",
    "kb+text",
    "kb+table",
    "kb+info",
    "text+table",
    "text+info",
    "table+info",
]

# Initialize dictionary for statistics
combination_stats = {combo: {"hit_count": 0} for combo in combinations}

hit_count = 0
hit_count_from_wrong_source = 0
total_count = len(question_ids_in_compmix)
hit_k = [1,3,5,10,20,50,100]

all_metrics = {
    'hit@1':0.0,
    'hit@3': 0.0,
    'hit@5': 0.0,
    'hit@10': 0.0,
    'hit@20': 0.0,
    'hit@50': 0.0,
    'hit@100': 0.0,
    'P@1':0.0,
    'P@3': 0.0,
    'P@5': 0.0,
    'P@10': 0.0,
    'P@20': 0.0,
    'P@50': 0.0,
    'P@100': 0.0,
    'mrr_score': 0.0,
    'inst_follow': 0.0
}

for item in tqdm(retrieved_evi_and_que):
    if item['question']['question_id'] not in question_ids_in_compmix:
        continue

    top_evidences_from_retrival = item['top_evidences']
    all_num = len(top_evidences_from_retrival)
    assert all_num == 100

    if use_single_source != 'all':
        top_evidences = [evidence for evidence in top_evidences_from_retrival if evidence["source"] == use_single_source]
        other_source_evi = [evidence for evidence in top_evidences_from_retrival if evidence["source"] != use_single_source]
    else:
        top_evidences = top_evidences_from_retrival
        other_source_evi = []

    inst_source_num = len(top_evidences)

    turn_answers = item['question']['answers']
    hit, answering_evidences = answer_presence(top_evidences, turn_answers)

    item["answer_presence"] = hit
    item["answer_presence_per_src"] = {
        evidence["source"]: 1 for evidence in answering_evidences
    }
    item["right_evi_right_source"] = answering_evidences

    hit_count += hit

    mrr_at_10, mrr_at_20, mrr_at_50, mrr_value = mrr_score(top_evidences, turn_answers)
    all_metrics["mrr_score"] += mrr_value
    all_metrics["inst_follow"] += inst_source_num/all_num

    prefix_sum = hit_at_100(top_evidences, turn_answers)
    for k in hit_k:
        k_value = 0.0

        if len(prefix_sum) == 0:
            break

        if k > len(prefix_sum):
            judge_val = prefix_sum[len(prefix_sum)-1]
        else:
            judge_val = prefix_sum[k-1]

        if judge_val > 0.0:
            k_value = 1.0

        k_sum = judge_val
        all_metrics[f"hit@{k}"] += k_value
        all_metrics[f"P@{k}"] += k_sum/k

    # Find the part that is not in answering_evidences
    not_answering_evidences = [evidence for evidence in top_evidences if evidence not in answering_evidences]
    item["wrong_evi_right_source"] = not_answering_evidences

    hit_wrong_source, answering_evidences_wrong_source = answer_presence(other_source_evi, turn_answers)
    item["answer_presence_from_wrong_src"] = {
        evidence["source"]: 1 for evidence in answering_evidences_wrong_source
    }
    item["right_evi_wrong_source"] = answering_evidences_wrong_source
    item["answer_presence_wrong_source"] = hit_wrong_source
    wrong_evi_wrong_source = [evidence for evidence in other_source_evi if evidence not in answering_evidences_wrong_source]
    item["wrong_evi_wrong_source"] = wrong_evi_wrong_source

    hit_count_from_wrong_source += hit_wrong_source

    # Delete the top_evidences attribute
    del item["top_evidences"]

    if hit == 1:
        sources = list(item["answer_presence_per_src"].keys())
        sources.sort()  # Sort to ensure consistency of combinations
        for combo in combinations:
            combo_sources = combo.split("+")
            if any(source in sources for source in combo_sources):
                combination_stats[combo]["hit_count"] += 1

print(f"retrieved_evi_and_que length: {total_count}")

precision_records = dict()
# Calculate the probability of hit==1 for each combination
for combo, stats in combination_stats.items():
    probability = stats["hit_count"] / total_count * 100.0
    print(f"{combo}: {probability:.2f}")
    precision_records[combo] = f"{probability:.2f}"

print(f"right source answer presence: {hit_count/total_count* 100.0:.2f}")
precision_records['right_source_All'] = f"{hit_count/total_count* 100.0:.2f}"

print(f"wrong source answer presence: {hit_count_from_wrong_source/total_count* 100.0:.2f}")
precision_records['wrong_source_All'] = f"{hit_count_from_wrong_source/total_count* 100.0:.2f}"

print()

for metric, value in all_metrics.items():
    value = value / total_count * 100.0
    precision_records[metric] = f"{value:.2f}"
    print(f"{metric}: {value:.2f}")

if convmix_test_set != 'test':
    output_path = f"eval_results/{model_name_base}_eval_result_{convmix_test_set}_{use_single_source}.json"
    with open(output_path, "w") as fp:
        json.dump(retrieved_evi_and_que, fp, indent=4)

output_path = f"eval_results/{model_name_base}_eval_metrics_{convmix_test_set}_{use_single_source}.json"
with open(output_path, "w") as fp:
    json.dump(precision_records, fp, indent=4)

print()
print('-'*15)
print()

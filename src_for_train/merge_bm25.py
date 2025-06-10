import json
import os

def merge_results(parts, output_file):
    merged_results = []

    for part in parts:
        part_file = f"retrival_results/bm25_retrived_evi_and_que_test_part{part}.json"
        with open(part_file, 'r') as f:
            part_results = json.load(f)
            merged_results.extend(part_results)

    with open(output_file, 'w') as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)

    print(f"Merged results saved to {output_file}")

if __name__ == "__main__":
    parts = [0, 1, 2, 3, 4]
    output_file = "retrival_results/bm25_retrived_evi_and_que_test_merged.json"
    merge_results(parts, output_file)


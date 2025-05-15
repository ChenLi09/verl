"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os
import hashlib
import re
from collections import defaultdict

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained("/home/share/data/model/Qwen2.5-7B", trust_remote_code=True)


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source1 = "DigitalLearningGmbH/MATH-lighteval"
    data_source2 = "qgallouedec/DAPO-Math-17k-Processed-Scored"
    
    dataset1 = datasets.load_dataset(data_source1, trust_remote_code=True)
    dataset2 = datasets.load_dataset(data_source2, trust_remote_code=True)

    train_dataset1 = dataset1["train"]
    train_dataset2 = dataset2["train"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = example.pop("problem") if data_source == data_source1 else example.pop("prompt")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer) if data_source == data_source1 else answer
            data = {
                "data_source": "math_agpo",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn
    
    def filter_by_length(example):
        prompt_tokens = tokenizer.encode(example["prompt"][0]["content"])
        return len(prompt_tokens) <= 2192

    def filter_math_level(example):
        level_str = example.get("level")
        if level_str and isinstance(level_str, str) and "Level " in level_str:
            try:
                level_num = int(level_str.split("Level ")[1])
                return level_num >= 3
            except ValueError:
                return False
        return False


    def filter_dapo_solve_rate(example):
        solve_rate = example.get("Qwen3-32B_solve_rate")
        if solve_rate is not None:
            return 0.5 <= solve_rate <= 0.8
        return False
    
    train_dataset1 = train_dataset1.filter(filter_math_level)
    train_dataset1 = train_dataset1.map(function=make_map_fn("train", data_source1), with_indices=True, remove_columns=dataset1["train"].column_names)
    train_dataset1 = train_dataset1.filter(filter_by_length)

    train_dataset2 = train_dataset2.filter(filter_dapo_solve_rate)
    train_dataset2 = train_dataset2.map(function=make_map_fn("train", data_source2), with_indices=True, remove_columns=dataset2["train"].column_names)
    train_dataset2 = train_dataset2.filter(filter_by_length)

    combined_train_dataset = datasets.concatenate_datasets([train_dataset1, train_dataset2])
    
    def deduplicate_dataset(dataset, method="exact"):
        """
        Deduplicate a dataset based on prompt content
        
        Args:
            dataset: The dataset to deduplicate
            method: The deduplication method
                - "exact": Exact string matching using hashes (default)
                - "normalized": Normalized string matching (lowercase, remove extra spaces)
        
        Returns:
            Deduplicated dataset
        """
        seen_hashes = set()
        indices_to_keep = []

        def get_hash(text):
            if method == "normalized":
                # Normalize: lowercase and remove extra whitespace
                text = re.sub(r'\s+', ' ', text.lower().strip())
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        prompt_to_idx = defaultdict(list)
        
        for idx, example in enumerate(dataset):
            prompt_content = example["prompt"][0]["content"]
            content_hash = get_hash(prompt_content)
            prompt_to_idx[content_hash].append(idx)
        
        for hash_value, idx_list in prompt_to_idx.items():
            indices_to_keep.append(idx_list[0])
        
        indices_to_keep.sort()
        
        total_examples = len(dataset)
        unique_examples = len(indices_to_keep)
        duplicates = total_examples - unique_examples
        
        print(f"Original dataset size: {total_examples}")
        print(f"After deduplication: {unique_examples}")
        print(f"Removed {duplicates} duplicates ({(duplicates/total_examples)*100:.2f}%)")
        
        return dataset.select(indices_to_keep)
    
    deduplicated_dataset = deduplicate_dataset(combined_train_dataset, method="normalized")
    
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    output_file = os.path.join(local_dir, "math_and_dapo_train_05_14.parquet")
    deduplicated_dataset.to_parquet(output_file)
    print(f"Deduplicated dataset saved to {output_file}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=output_file, dst=args.hdfs_dir)
        print(f"Copied deduplicated dataset to HDFS directory: {args.hdfs_dir}")

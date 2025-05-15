# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

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
    data_source1 = "HuggingFaceH4/MATH-500"
    data_source2 = "reasoning-machines/gsm-hard"
    dataset1 = datasets.load_dataset(data_source1, trust_remote_code=True)
    dataset2 = datasets.load_dataset(data_source2, trust_remote_code=True)

    test_dataset1 = dataset1["test"]
    test_dataset2 = dataset2["train"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = example.pop("problem") if data_source == data_source1 else example.pop("input")

            question = question + " " + instruction_following

            if data_source == data_source1:
                answer = example.pop("solution")
                solution = extract_solution(answer)
            else:
                target = example.pop("target")
                answer = str(int(target)) if float(target).is_integer() else str(target)
                solution = answer
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn
    
    def filter_by_length(example):
        prompt_tokens = tokenizer.encode(example["prompt"][0]["content"])
        return len(prompt_tokens) <= 2192  # Common max length threshold
        
    test_dataset1 = test_dataset1.map(function=make_map_fn("test", data_source1), with_indices=True)
    test_dataset1 = test_dataset1.filter(filter_by_length)

    test_dataset2 = test_dataset2.map(function=make_map_fn("train", data_source2), with_indices=True)
    test_dataset2 = test_dataset2.filter(filter_by_length)

    test_dataset = datasets.concatenate_datasets([test_dataset1, test_dataset2])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    print(f"test_dataset length: {len(test_dataset)}")

    test_dataset.to_parquet(os.path.join(local_dir, "gsm_hard_math_500_test_05_14.parquet"))

    for i in range(10):
        print(test_dataset[1000 + i])

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

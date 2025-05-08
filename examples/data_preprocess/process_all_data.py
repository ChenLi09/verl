from datasets import load_dataset
import json
import os
import re
from tqdm import tqdm
import hashlib
from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("/home/yangkai/models/Qwen2.5-32B-Instruct")

# Create output directory
output_dir = "/home/yangkai/data/data_process"
os.makedirs(output_dir, exist_ok=True)

# Path for the final processed data
orz_math_path = "/home/yangkai/data/data_process/orz_math_13k_collection_hard.json"
additional_dataset_path = "/home/yangkai/data/data_process/hard_problems_with_rate.jsonl"
merged_data_path = "/home/yangkai/data/data_process/final_merged_math_data.jsonl"

big_math_rl_processed_path = os.path.join(output_dir, "big_math_rl_filtered.jsonl")
dapo_math_processed_path = os.path.join(output_dir, "dapo_math_filtered.jsonl")
skywork_math_processed_path = os.path.join(output_dir, "skywork_math_filtered.jsonl")

# Function to create a hash of the question content for deduplication
def get_question_hash(question):
    # Normalize the question by removing extra whitespace and converting to lowercase
    normalized_question = ' '.join(question.strip().lower().split())
    return hashlib.md5(normalized_question.encode('utf-8')).hexdigest()

# Function to check token length using the specified tokenizer
def check_token_lengths(question, answer, max_question_tokens=2096, max_answer_tokens=100):
    if not question or not answer:
        # If either question or answer is empty, skip this item
        return False
    
    # Check question length
    question_encoding = tokenizer(question, return_attention_mask=False)
    question_tokens = len(question_encoding.input_ids)
    if question_tokens > max_question_tokens:
        return False

    # Check answer length if present
    if answer:
        answer_encoding = tokenizer(answer, return_attention_mask=False)
        answer_tokens = len(answer_encoding.input_ids)
        if answer_tokens > max_answer_tokens:
            return False

    return True

def format_question(question_text):
    instruction = "Please reason step by step and output the final answer within \\boxed{}.\n\n"
    
    formatted_content = question_text.strip() + instruction
    
    # Create the conversation format
    conversation = [{"content": formatted_content, "role": "user"}]
    
    return json.dumps(conversation)

print("Processing datasets...")

# Process the dataset
filtered_count = 0
token_filtered_count = 0
total_count = 0
all_items = []
seen_questions = set()

# Process the Orz Math JSON data
print(f"Processing orz_math_13k data from {orz_math_path}...")
orz_count = 0
orz_filtered = 0
orz_token_filtered = 0

try:
    with open(orz_math_path, 'r', encoding='utf-8') as infile:
        orz_data = json.load(infile)
        orz_count = len(orz_data)
        
        for item in tqdm(orz_data):
            # Extract question from human message
            if len(item) >= 2 and isinstance(item[0], dict) and item[0].get("from") == "human":
                question = item[0].get("value", "")
                
                # Extract answer from assistant message
                answer = ""
                if len(item) >= 2 and isinstance(item[1], dict) and item[1].get("from") == "assistant":
                    if "ground_truth" in item[1] and isinstance(item[1]["ground_truth"], dict):
                        answer = item[1]["ground_truth"].get("value", "")

                question  = format_question(question)
                
                # Skip if token length exceeds limits
                if not check_token_lengths(question, answer):
                    orz_token_filtered += 1
                    continue
                
                # Create standardized item
                if question:
                    standardized_item = {
                        "question": question,
                        "answer": answer,
                        "extra_params": {
                            "level": 3,
                            "source": "orz_math_data"
                        }
                    }
                    
                    # Get hash for deduplication
                    question_hash = get_question_hash(question)
                    
                    # Add if not a duplicate
                    if question_hash not in seen_questions:
                        seen_questions.add(question_hash)
                        all_items.append(standardized_item)
                        orz_filtered += 1
except Exception as e:
    print(f"Error processing orz_math data: {e}")

print(f"Orz Math dataset: processed {orz_filtered} unique items out of {orz_count} total")
print(f"Skipped {orz_token_filtered} examples due to token length limits.")

# Now read the additional dataset and merge
print(f"Loading and merging additional dataset from {additional_dataset_path}...")
add_total = 0
add_filtered = 0
add_token_filtered = 0

try:
    with open(additional_dataset_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile):
            add_total += 1
            try:
                item = json.loads(line)
                
                # Extract question and reference_answer, adjusting field names if necessary
                question = item.get("question", "")
                if not question:
                    question = item.get("problem", "")  # Alternative field name
                    
                answer = item.get("reference_answer", "")
                if not answer:
                    answer = item.get("answer", "")  # Alternative field name
                
                if not question:
                    continue
                
                # Get accuracy/solve rate if available
                accuracy = item.get("accuracy", None) or item.get("solve_rate", None)
                
                # Determine level based on accuracy
                level = None
                if accuracy is None:
                    continue
                if 0.3 <= accuracy <= 0.4:
                    level = 1
                elif accuracy < 0.1:
                    level = 3
                elif 0.1 <= accuracy <= 0.2:
                    level = 2
                else:
                    # Skip items that don't match our level criteria
                    continue
                
                question = format_question(question)

                # Skip if token length exceeds limits
                if not check_token_lengths(question, answer):
                    add_token_filtered += 1
                    continue
                    
                # Create consistent format with extra_params
                standardized_item = {
                    "question": question,
                    "answer": answer,
                    "extra_params": {
                        "level": level,
                        "source": "general_reasoning",
                    }
                }
                
                # Get hash for deduplication
                question_hash = get_question_hash(question)
                
                # Add if not a duplicate
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    all_items.append(standardized_item)
                    add_filtered += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in additional dataset")
                continue
except FileNotFoundError:
    print(f"Warning: Additional dataset file not found at {additional_dataset_path}")

# Process Big-Math-RL-Verified dataset
print("Loading Big-Math-RL-Verified dataset...")
print("Note: This dataset is gated. Make sure you're authenticated with `huggingface-cli login`")
big_math_total = 0
big_math_filtered_by_rate = 0
big_math_unique = 0
big_math_token_filtered = 0

try:
    # Load the dataset with authentication token
    # This assumes you've already run `huggingface-cli login` or similar
    big_math_ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified")
    
    # Process the dataset and filter by llama8b_solve_rate
    with open(big_math_rl_processed_path, 'w', encoding='utf-8') as outfile:
        # Process each split in the dataset
        for split in big_math_ds.keys():
            print(f"Processing Big-Math-RL split: {split}")
            for item in tqdm(big_math_ds[split]):
                big_math_total += 1
                
                # Extract relevant fields
                problem = item.get("problem", "")
                answer = item.get("answer", "")
                solution = item.get("solution", "")
                solve_rate = item.get("llama8b_solve_rate")
                
                # Skip if the item doesn't have llama8b_solve_rate or it's not <= 0.4
                if solve_rate is None or solve_rate > 0.2:
                    continue
                
                problem = format_question(problem)

                # Skip if token length exceeds limits
                if not check_token_lengths(problem, answer):
                    big_math_token_filtered += 1
                    continue
                
                # Determine level based on solve rate
                level = None
                if 0.15 <= solve_rate <= 0.2:
                    level = 1
                    big_math_filtered_by_rate += 1
                elif solve_rate < 0.1:
                    level = 3
                    big_math_filtered_by_rate += 1
                elif 0.1 <= solve_rate < 0.15:
                    level = 2
                    big_math_filtered_by_rate += 1
                
                if level is not None:
                    # Create standardized item format with extra_params
                    standardized_item = {
                        "question": problem,
                        "answer": answer,
                        "extra_params": {
                            "level": level,
                            "source": "big_math_rl_verified",
                        }
                    }
                    
                    # Get hash for deduplication
                    question_hash = get_question_hash(problem)
                    
                    # Add if not a duplicate
                    if question_hash not in seen_questions:
                        seen_questions.add(question_hash)
                        all_items.append(standardized_item)
                        outfile.write(json.dumps(standardized_item) + "\n")
                        big_math_unique += 1
    
    print(f"Big-Math-RL processing complete:")
    print(f"Total examples processed: {big_math_total}")
    print(f"Examples filtered by solve rate criteria: {big_math_filtered_by_rate}")
    print(f"Skipped {big_math_token_filtered} examples due to token length limits.")
    print(f"Unique examples after deduplication: {big_math_unique}")
    print(f"Filtered data saved to {big_math_rl_processed_path}")
except Exception as e:
    print(f"Error processing Big-Math-RL dataset: {e}")
    print("If this is an authentication error, please run: huggingface-cli login")
    print("Then enter your Hugging Face access token when prompted")

# Process DAPO-Math-17k dataset
print("Loading DAPO-Math-17k dataset...")
dapo_math_total = 0
dapo_math_filtered_by_rate = 0
dapo_math_unique = 0
dapo_math_token_filtered = 0

try:
    # Load the dataset
    dapo_math_ds = load_dataset("qgallouedec/DAPO-Math-17k-Processed-Scored")
    
    # Process the dataset and filter by Qwen3-32B_solve_rate
    with open(dapo_math_processed_path, 'w', encoding='utf-8') as outfile:
        # Process each split in the dataset
        for split in dapo_math_ds.keys():
            print(f"Processing DAPO-Math split: {split}")
            for item in tqdm(dapo_math_ds[split]):
                dapo_math_total += 1
                
                # Extract relevant fields
                prompt = item.get("prompt", "")
                solution = item.get("solution", "")  # This is the answer
                data_source = item.get("data_source", "")
                solve_rate = item.get("Qwen3-32B_solve_rate")
                
                # Skip if the item doesn't have Qwen3-32B_solve_rate or it's > 0.5
                if solve_rate is None or solve_rate > 0.5:
                    continue
                
                # Format question with instructions
                question = format_question(prompt)

                # Skip if token length exceeds limits
                if not check_token_lengths(question, solution):
                    dapo_math_token_filtered += 1
                    continue
                
                # Determine level based on solve rate
                level = None
                if 0.4 <= solve_rate <= 0.5:
                    level = 1
                    dapo_math_filtered_by_rate += 1
                elif 0.2 <= solve_rate < 0.4:
                    level = 2
                    dapo_math_filtered_by_rate += 1
                elif solve_rate < 0.2:
                    level = 3
                    dapo_math_filtered_by_rate += 1
                else:
                    # Should not happen given our filtering, but just in case
                    continue
                
                # Create standardized item format with extra_params
                standardized_item = {
                    "question": question,
                    "answer": solution,
                    "extra_params": {
                        "level": level,
                        "source": data_source,
                    }
                }
                
                # Get hash for deduplication
                question_hash = get_question_hash(prompt)  # Use original prompt for hashing
                
                # Add if not a duplicate
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    all_items.append(standardized_item)
                    outfile.write(json.dumps(standardized_item) + "\n")
                    dapo_math_unique += 1
    
    print(f"DAPO-Math processing complete:")
    print(f"Total examples processed: {dapo_math_total}")
    print(f"Examples filtered by solve rate criteria: {dapo_math_filtered_by_rate}")
    print(f"Skipped {dapo_math_token_filtered} examples due to token length limits.")
    print(f"Unique examples after deduplication: {dapo_math_unique}")
    print(f"Filtered data saved to {dapo_math_processed_path}")
except Exception as e:
    print(f"Error processing DAPO-Math dataset: {e}")

# Process Skywork/Skywork-OR1-RL-Data dataset
print("Loading Skywork/Skywork-OR1-RL-Data dataset...")
skywork_math_total = 0
skywork_math_filtered_by_rate = 0
skywork_math_unique = 0
skywork_math_token_filtered = 0

try:
    # Load the dataset
    skywork_math_ds = load_dataset("Skywork/Skywork-OR1-RL-Data")
    
    # Process the dataset and filter by solve_rate
    with open(skywork_math_processed_path, 'w', encoding='utf-8') as outfile:
        # Process each split in the dataset
        for split in skywork_math_ds.keys():
            if split != "math":
                continue

            print(f"Processing Skywork Math split: {split}")
            for item in tqdm(skywork_math_ds[split]):
                skywork_math_total += 1
                
                # Extract relevant fields
                question = item.get("prompt", "")[0]["content"]
                answer = item.get("reward_model", "")["ground_truth"][0]
                model_difficulty = item.get("extra_info", None)["model_difficulty"]["DeepSeek-R1-Distill-Qwen-7B"]
                
                
                
                if model_difficulty is None or model_difficulty < 6:
                    continue
                
                # Skip if token length exceeds limits
                if not check_token_lengths(question, answer):
                    skywork_math_token_filtered += 1
                    continue
                
                # Determine level based on solve rate
                level = None
                if 6 <= model_difficulty <= 11:
                    level = 1
                    skywork_math_filtered_by_rate += 1
                elif 12 <= model_difficulty <= 14:
                    level = 2
                    skywork_math_filtered_by_rate += 1
                elif 15 <= model_difficulty <= 16:
                    level = 3
                    skywork_math_filtered_by_rate += 1
                else:
                    # Should not happen given our filtering, but just in case
                    continue
                
                # Create standardized item format with extra_params
                standardized_item = {
                    "question": format_question(question),
                    "answer": answer,
                    "extra_params": {
                        "level": level,
                        "source": "skywork_math",
                    }
                }
                
                # Get hash for deduplication - use the original question text for hashing
                question_hash = get_question_hash(question_text if 'question_text' in locals() else question)
                
                # Add if not a duplicate
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    all_items.append(standardized_item)
                    outfile.write(json.dumps(standardized_item) + "\n")
                    skywork_math_unique += 1
    
    print(f"Skywork Math processing complete:")
    print(f"Total examples processed: {skywork_math_total}")
    print(f"Examples filtered by solve rate criteria: {skywork_math_filtered_by_rate}")
    print(f"Skipped {skywork_math_token_filtered} examples due to token length limits.")
    print(f"Unique examples after deduplication: {skywork_math_unique}")
    print(f"Filtered data saved to {skywork_math_processed_path}")
except Exception as e:
    print(f"Error processing Skywork Math dataset: {e}")

# Write the final merged and deduplicated dataset
with open(merged_data_path, 'w', encoding='utf-8') as outfile:
    for item in all_items:
        outfile.write(json.dumps(item) + "\n")

# Print summary of processed datasets
print(f"Additional dataset: processed {add_filtered} unique items out of {add_total} total (skipped {add_token_filtered} due to token length)")
print(f"Orz Math dataset: processed {orz_filtered} unique items out of {orz_count} total (skipped {orz_token_filtered} due to token length)")
if 'big_math_unique' in locals():
    print(f"Big-Math-RL dataset: processed {big_math_unique} unique items with appropriate solve rates out of {big_math_total} total (skipped {big_math_token_filtered} due to token length)")
if 'dapo_math_unique' in locals():
    print(f"DAPO-Math dataset: processed {dapo_math_unique} unique items with appropriate solve rates out of {dapo_math_total} total (skipped {dapo_math_token_filtered} due to token length)")
if 'skywork_math_unique' in locals():
    print(f"Skywork Math dataset: processed {skywork_math_unique} unique items with appropriate solve rates out of {skywork_math_total} total (skipped {skywork_math_token_filtered} due to token length)")
print(f"Final merged and deduplicated dataset contains {len(all_items)} items")
print(f"Final merged data saved to {merged_data_path}")
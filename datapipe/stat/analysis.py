from collections import Counter
from datasets import Dataset

file_path = "/home/share/reasoning/rl_math_data.parquet"

train_dataset = Dataset.from_parquet(file_path)

# print level and source distribution
level_counter = Counter()
source_counter = Counter()
math_category_counter = Counter()

for example in train_dataset:
    level_counter[example['extra_info']['level']] += 1
    source_counter[example['extra_info']['original_source']] += 1
    math_category_counter[example['extra_info']['math_category']] += 1

# print level distribution
print(f"{'Level':<8}{'Count':<10}{'Percentage':<10}")
for level in sorted(level_counter.keys()):
    count = level_counter[level]
    percent = (count / len(train_dataset)) * 100
    print(f"{level:<8}{count:<10}{percent:.2f}%")

# print source distribution
print(f"{'Source':<25}{'Count':<10}{'Percentage':<10}")
for source in sorted(source_counter.keys()):
    count = source_counter[source]
    percent = (count / len(train_dataset)) * 100
    print(f"{source:<25}{count:<10}{percent:.2f}%")

# print math category distribution
print(f"{'Math Category':<20}{'Count':<10}{'Percentage':<10}")
for category in sorted(math_category_counter.keys()):
    count = math_category_counter[category]
    percent = (count / len(train_dataset)) * 100
    print(f"{category:<20}{count:<10}{percent:.2f}%")

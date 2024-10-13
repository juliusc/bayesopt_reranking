import json
import random

def sample_jsonl_by_language_pair(input_file, output_file, num_samples=20):
    random.seed(41)
    # Dictionary to store instances per language pair
    language_pairs = {}

    # Read the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            instance = json.loads(line.strip())
            langs = instance.get("langs")
            if langs not in language_pairs:
                language_pairs[langs] = []
            language_pairs[langs].append(instance)

    # Print the language pairs found
    print("Language pairs found in the dataset:")
    for langs in language_pairs.keys():
        print(f"- {langs}")

    # Sample 20 instances per language pair
    sampled_data = []
    for langs, instances in language_pairs.items():
        if len(instances) > num_samples:
            sampled_data.extend(random.sample(instances, num_samples))
        else:
            sampled_data.extend(instances)  # If there are fewer than 20, include all

    # Write the sampled instances to a new JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for instance in sampled_data:
            outfile.write(json.dumps(instance, ensure_ascii=False) + '\n')

    print(f"Sampled {num_samples} instances per language pair and wrote to {output_file}.")


# Usage
input_file = 'test_small.jsonl'  # Replace with your input JSONL file path
output_file = 'runtime_sample2.jsonl'  # Replace with your output JSONL file path
sample_jsonl_by_language_pair(input_file, output_file, num_samples=20)

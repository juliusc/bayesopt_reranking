import json

def sample_shortest_by_language_pair(input_file, output_file, num_samples=2):
    # Dictionary to store instances per language pair
    language_pairs = {}

    # Read the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            instance = json.loads(line.strip())
            langs = instance.get("langs")
            text_length = len(instance.get("text", ""))  # Get the length of the "text" field
            if langs not in language_pairs:
                language_pairs[langs] = []
            language_pairs[langs].append((text_length, instance))

    # Print the language pairs found
    print("Language pairs found in the dataset:")
    for langs in language_pairs.keys():
        print(f"- {langs}")

    # Sample the 2 shortest instances per language pair
    sampled_data = []
    for langs, instances in language_pairs.items():
        # Sort the instances by the length of the "text" field
        instances_sorted = sorted(instances, key=lambda x: x[0])
        # Select the 2 shortest instances (or fewer if not enough)
        sampled_data.extend([inst[1] for inst in instances_sorted[:num_samples]])

    # Write the sampled instances to a new JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for instance in sampled_data:
            outfile.write(json.dumps(instance, ensure_ascii=False) + '\n')

    print(f"Sampled {num_samples} shortest instances per language pair and wrote to {output_file}.")


# Usage
input_file = 'test_small.jsonl'  # Replace with your input JSONL file path
output_file = 'shortest_sample.jsonl'  # Replace with your output JSONL file path
sample_shortest_by_language_pair(input_file, output_file, num_samples=2)

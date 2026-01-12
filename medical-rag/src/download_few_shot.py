"""Download real few-shot examples from PubMedQA artificial split.

Uses the pqa_artificial split from HuggingFace (211k machine-generated Q&A pairs)
to get diverse, real examples for few-shot prompting.
"""

import json
from pathlib import Path

from datasets import load_dataset


def download_few_shot_examples(
    num_per_type: int = 2,
    output_path: str = "data/pubmedqa/few_shot_examples.json"
):
    """Download real few-shot examples from PubMedQA artificial split.

    Args:
        num_per_type: Number of examples per answer type (yes/no/maybe)
        output_path: Where to save the JSON file
    """
    print("Loading PubMedQA pqa_artificial split from HuggingFace...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    print(f"Loaded {len(dataset)} examples")

    # Collect examples for each answer type
    examples = {
        "yes": [],
        "no": [],
        "maybe": []
    }

    for item in dataset:
        answer = item['final_decision'].lower()
        if answer not in examples:
            continue

        if len(examples[answer]) < num_per_type:
            # Join context paragraphs into single string
            context = " ".join(item['context']['contexts'])

            examples[answer].append({
                "pubid": item['pubid'],
                "question": item['question'],
                "context": context,
                "answer": answer,
                "long_answer": item['long_answer']
            })

        # Check if we have enough examples
        if all(len(v) >= num_per_type for v in examples.values()):
            break

    # Summary
    print("\nCollected examples:")
    for answer_type, items in examples.items():
        print(f"  {answer_type}: {len(items)} examples")

    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"\nSaved to: {output_file}")

    # Print sample
    print("\n" + "="*60)
    print("Sample example (first 'yes' example):")
    print("="*60)
    sample = examples['yes'][0]
    print(f"Question: {sample['question']}")
    print(f"Context (truncated): {sample['context'][:300]}...")
    print(f"Answer: {sample['answer']}")

    return examples


if __name__ == "__main__":
    download_few_shot_examples()

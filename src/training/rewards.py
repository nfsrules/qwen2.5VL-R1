import re

def extract_answer(text):
    """Extracts content between <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def reward_correct_answer(samples, **kwargs):
    rewards = []
    for sample in samples:
        # Reconstruct the full output (model's generation)
        conversation = " ".join(turn["value"] for turn in sample["conversations"])
        predicted_answer = extract_answer(conversation)

        # Ground truth is the final assistant turn
        ground_truth = sample["conversations"][-1]["value"]
        expected_answer = extract_answer(ground_truth)

        if predicted_answer is None or expected_answer is None:
            rewards.append(0.0)
        elif predicted_answer == expected_answer:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

import re

def extract_answer(text):
    """Extracts content between <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def reward_correct_answer(prompts=None, completions=None, **kwargs):
    """
    Compares extracted <answer>...</answer> from completions vs ground truth in prompts
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        expected = extract_answer(prompt)
        predicted = extract_answer(completion)

        if predicted is None or expected is None:
            rewards.append(0.0)
        elif predicted == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


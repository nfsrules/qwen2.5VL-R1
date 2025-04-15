import re


def extract_think(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def flatten_prompt(prompt):
    if isinstance(prompt, list):
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in prompt)
    return prompt


def match_format_exactly(prompts=None, completions=None, **kwargs):
    scores = []
    for i, c in enumerate(completions):
        c_str = flatten_prompt(c)
        has_think = "<think>" in c_str and "</think>" in c_str
        has_answer = "<answer>" in c_str and "</answer>" in c_str
        score = 3.0 if has_think and has_answer else 0.0
        print(
            f"[ExactFormat] Ex {i} | has_think: {has_think}, has_answer: {has_answer} → score: {score}"
        )
        scores.append(score)
    return scores


def match_format_approximately(prompts=None, completions=None, **kwargs):
    scores = []
    for i, c in enumerate(completions):
        c_str = flatten_prompt(c)
        score = 0.0
        score += 0.5 if c_str.count("<think>") == 1 else -0.5
        score += 0.5 if c_str.count("</think>") == 1 else -0.5
        score += 0.5 if c_str.count("<answer>") == 1 else -0.5
        score += 0.5 if c_str.count("</answer>") == 1 else -0.5
        print(f"[ApproxFormat] Ex {i} | score: {score}")
        scores.append(score)
    return scores


def check_answer(prompts=None, completions=None, **kwargs):
    scores = []

    print("prompts", prompts)
    print("completions", completions)

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        prompt_str = flatten_prompt(prompt)
        completion_str = flatten_prompt(completion)

        expected = extract_answer(prompt_str)
        predicted = extract_answer(completion_str)

        if not predicted:
            match = re.search(r"\(([A-D])\)", completion_str)
            if match:
                predicted = f"({match.group(1)})"
            else:
                predicted = None

        if not expected:
            match = re.search(r"\(([A-D])\)", prompt_str)
            if match:
                expected = f"({match.group(1)})"
            else:
                expected = None

        print(f"[CheckAnswer] Ex {i} | Expected: {expected} | Predicted: {predicted}")

        if not predicted or not expected:
            score = 0.0
            print(f"[CheckAnswer] Ex {i} → score: {score} (missing)")
        elif predicted.strip().lower() == expected.strip().lower():
            score = 3.0
            print(f"[CheckAnswer] Ex {i} → score: {score} (match)")
        else:
            score = -1.0
            print(f"[CheckAnswer] Ex {i} → score: {score} (mismatch)")

        scores.append(score)

    return scores


def check_reasoning_length(prompts=None, completions=None, **kwargs):
    scores = []
    for i, c in enumerate(completions):
        c_str = flatten_prompt(c)
        reasoning = extract_think(c_str)
        word_count = len(reasoning.split()) if reasoning else 0
        score = 0.5 if word_count >= 10 else 0.0
        print(f"[ReasoningLen] Ex {i} | Word count: {word_count} → score: {score}")
        scores.append(score)
    return scores

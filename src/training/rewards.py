import re


def extract_answer(text):
    print()
    if not text:
        return None

    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"\(([A-D])\)", text)
    if match:
        return f"({match.group(1)})"

    return None


def flatten_prompt(prompt):
    if isinstance(prompt, list):
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in prompt)
    return prompt


def check_answer(prompts=None, completions=None, **kwargs):
    scores = []

    answers = kwargs.get("answer")
    
    for i in range(len(prompts)):
        prompt = prompts[i]
        completion = completions[i]
        answer_ref = answers[i] if answers and i < len(answers) else None

        prompt_str = flatten_prompt(prompt)
        completion_str = flatten_prompt(completion)

        expected = extract_answer(answer_ref) if answer_ref else None
        predicted = extract_answer(completion_str)

        if not predicted:
            match = re.search(r"\(([A-D])\)", completion_str)
            if match:
                predicted = f"({match.group(1)})"

        if not predicted or not expected:
            score = -1.0
            print(f"[CheckAnswer] Ex {i} → score: {score} (missing)", flush=True)
        elif predicted.strip().lower() == expected.strip().lower():
            score = 3.0
            print(f"[CheckAnswer] Ex {i} → score: {score} (match)", flush=True)
        else:
            score = -1.0
            print(f"[CheckAnswer] Ex {i} → score: {score} (mismatch)", flush=True)

        scores.append(score)

    return scores

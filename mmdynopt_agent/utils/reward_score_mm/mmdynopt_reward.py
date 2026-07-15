import re
import unicodedata
from typing import Dict, Optional, Union

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_answer(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def cal_f1_score(prediction, ground_truth):
    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    else:
        ground_truth_list = ground_truth

    max_f1 = 0.0
    pred_tokens = normalize_answer(prediction).split()

    for gt in ground_truth_list:
        gold_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        max_f1 = max(max_f1, f1)

    return max_f1


def exact_match_score(prediction, ground_truth):
    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    else:
        ground_truth_list = ground_truth

    normalized_prediction = normalize_answer(prediction)

    for gt in ground_truth_list:
        if normalize_answer(gt) == normalized_prediction:
            return 1.0

    return 0.0


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1.0
    return 0.0


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return 1.0
    return 0.0


def extract_solution(solution_str):
    if not solution_str:
        return None
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def has_answer_tags(solution_str):
    if solution_str is None:
        return False
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match and match.group(1).strip():
        return True
    return False


def _convert_to_string(solution_str):
    if solution_str is None:
        return ""
    if isinstance(solution_str, list):
        if len(solution_str) == 0:
            return ""

        return '\n'.join(str(s) for s in solution_str if s)
    return str(solution_str)


def compute_score_format(solution_str):
    if solution_str is None:
        return 0.0

    try:
        format_reward = 0.0

        if isinstance(solution_str, list):
            assistant_blocks = [s for s in solution_str if s]
        else:
            assistant_blocks = [solution_str] if solution_str else []

        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0

        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if (assistant_block.count('<redacted_thinking>') == 1 and
                assistant_block.count('</redacted_thinking>') == 1 and
                assistant_block.count('<prompt>') == 1 and
                assistant_block.count('</prompt>') == 1):
                think_match = re.search(
                    r'^<redacted_thinking>(.*?)</redacted_thinking>(\s*)<prompt>(.*?)</prompt>$',
                    assistant_block,
                    re.DOTALL
                )
                if think_match:
                    format_reward += 0.3

        last_assistant_block = assistant_blocks[-1]

        if has_answer_tags(last_assistant_block):
            format_reward += 0.15

            answer_content = extract_solution(last_assistant_block)
            if answer_content and len(answer_content.strip()) > 0:
                format_reward += 0.15

            think_answer_match = re.search(
                r'^<redacted_thinking>(.*?)</redacted_thinking>(.*?)<answer>(.*?)</answer>$',
                last_assistant_block,
                re.DOTALL
            )
            if think_answer_match:
                format_reward += 0.1

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0

    return format_reward


def compute_score_answer(solution_str, ground_truth):
    if solution_str is None or not ground_truth:
        return 0.0

    try:
        if isinstance(solution_str, list):
            if len(solution_str) == 0:
                return 0.0
            last_block = solution_str[-1]
        else:
            last_block = solution_str

        if not has_answer_tags(last_block):
            return 0.0

        answer = extract_solution(last_block)
        if answer is None:
            return 0.0

        return cal_f1_score(answer, ground_truth)

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0


def compute_score_sm(solution_str, ground_truth):
    if solution_str is None or not ground_truth:
        return 0.0

    try:
        if isinstance(solution_str, list):
            if len(solution_str) == 0:
                return 0.0
            last_block = solution_str[-1]
        else:
            last_block = solution_str

        answer = extract_solution(last_block)
        if answer is None:
            return 0.0

        return float(subem_check(answer, ground_truth))

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_sm: {e}")
        return 0.0


def compute_score_em(solution_str, ground_truth):
    if solution_str is None or not ground_truth:
        return 0.0

    try:
        if isinstance(solution_str, list):
            if len(solution_str) == 0:
                return 0.0
            last_block = solution_str[-1]
        else:
            last_block = solution_str

        answer = extract_solution(last_block)
        if answer is None:
            return 0.0

        return exact_match_score(answer, ground_truth)

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_em: {e}")
        return 0.0


def compute_score_f1(solution_str, ground_truth):
    if solution_str is None or not ground_truth:
        return 0.0

    try:
        if isinstance(solution_str, list):
            if len(solution_str) == 0:
                return 0.0
            last_block = solution_str[-1]
        else:
            last_block = solution_str

        answer = extract_solution(last_block)
        if answer is None:
            return 0.0

        return cal_f1_score(answer, ground_truth)

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_f1: {e}")
        return 0.0


def compute_score_format_answer(solution_str, ground_truth, extra_info=None):
    if solution_str is None or not ground_truth:
        return 0.0

    if isinstance(solution_str, list) and len(solution_str) == 0:
        return 0.0
    if isinstance(solution_str, str) and solution_str == "":
        return 0.0

    try:

        format_reward = compute_score_format(solution_str)

        answer_reward = compute_score_answer(solution_str, ground_truth)

        budget_score = compute_score_budget_penalty(solution_str, extra_info)

        format_reward = min(format_reward, 1.0)

        if format_reward == 1.0:

            base = -1.0 + format_reward + answer_reward

            lambda_b = 0.1
            bonus = answer_reward * lambda_b * budget_score if answer_reward > 0.8 else 0.0

            return base + bonus
        else:

            return -1.0 + format_reward

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0


def compute_score_budget_penalty(solution_str, extra_info: Optional[Dict] = None) -> float:

    if extra_info and isinstance(extra_info, dict):
        if 'llm_prompt_len' in extra_info and 'llm_response_len' in extra_info:
            prompt_len = extra_info['llm_prompt_len']
            response_len = extra_info['llm_response_len']
            n_turns = extra_info.get('n_llm_calls', 0)

            prompt_tokens = int(prompt_len * 0.4)
            response_tokens = int(response_len * 0.4)

            lambda_input = 1.0
            lambda_output = 4.0
            lambda_turn = 0.05

            token_cost = lambda_input * prompt_tokens + lambda_output * response_tokens
            turn_cost = lambda_turn * max(0, n_turns - 1)
            total_cost = token_cost + turn_cost

            max_turns = float(extra_info.get("budget_max_turns", 5))
            max_prompt_tokens = float(extra_info.get("budget_max_prompt_tokens", 500))
            max_response_tokens = float(extra_info.get("budget_max_response_tokens", 1000))

            turn_frac = 0.0 if max_turns <= 0 else min(n_turns / max_turns, 1.0)
            prompt_frac = 0.0 if max_prompt_tokens <= 0 else min(prompt_tokens / max_prompt_tokens, 1.0)
            response_frac = 0.0 if max_response_tokens <= 0 else min(response_tokens / max_response_tokens, 1.0)

            token_frac = 0.5 * prompt_frac + 0.5 * response_frac

            cost = 0.5 * turn_frac + 0.5 * token_frac
            score = 1.0 - cost

            return max(0.0, min(1.0, score))

    if solution_str is None:
        return 0.0
    if isinstance(solution_str, list) and len(solution_str) == 0:
        return 0.0
    if isinstance(solution_str, str) and solution_str == "":
        return 0.0

    try:

        if isinstance(solution_str, list):
            n_turns = max(0, len(solution_str) - 1)
            all_prompts = []
            for block in solution_str:
                prompts_in_block = re.findall(r"<prompt>(.*?)</prompt>", block, re.DOTALL)
                all_prompts.extend(prompts_in_block)
        else:
            all_prompts = re.findall(r"<prompt>(.*?)</prompt>", solution_str, re.DOTALL)
            n_turns = len(all_prompts)

        prompt_tokens = int(len(''.join(all_prompts)) * 0.4) if all_prompts else 0

        lambda_input = 1.0
        lambda_turn = 0.05
        token_cost = lambda_input * prompt_tokens
        turn_cost = lambda_turn * max(0, n_turns - 1)
        total_cost = token_cost + turn_cost

        extra = extra_info or {}
        max_turns = float(extra.get("budget_max_turns", 5))
        max_prompt_tokens = float(extra.get("budget_max_prompt_tokens", 500))

        turn_frac = 0.0 if max_turns <= 0 else min(n_turns / max_turns, 1.0)
        prompt_frac = 0.0 if max_prompt_tokens <= 0 else min(prompt_tokens / max_prompt_tokens, 1.0)

        cost = 0.5 * turn_frac + 0.5 * prompt_frac
        score = 1.0 - cost

        return max(0.0, min(1.0, score))

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_budget_penalty: {e}")
        return 0.0

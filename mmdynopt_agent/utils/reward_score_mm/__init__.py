from . import mmdynopt_reward


def _default_compute_score_format(data_source, solution_str, extra_info=None):
    res = mmdynopt_reward.compute_score_format(solution_str)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info=None):
    res = mmdynopt_reward.compute_score_em(solution_str, ground_truth)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_sm(data_source, solution_str, ground_truth, extra_info=None):
    res = mmdynopt_reward.compute_score_sm(solution_str, ground_truth)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_f1(data_source, solution_str, ground_truth, extra_info=None):
    res = mmdynopt_reward.compute_score_f1(solution_str, ground_truth)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    res = mmdynopt_reward.compute_score_format_answer(solution_str, ground_truth, extra_info)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_budget(data_source, solution_str, ground_truth, extra_info=None):
    res = mmdynopt_reward.compute_score_budget_penalty(solution_str, extra_info)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info)


def _compute_score_detailed(data_source, solution_str, ground_truth, extra_info=None):
    return {
        "score": _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info),
        "em": _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info),
        "sm": _default_compute_score_sm(data_source, solution_str, ground_truth, extra_info),
        "f1": _default_compute_score_f1(data_source, solution_str, ground_truth, extra_info),
        "format": _default_compute_score_format(data_source, solution_str, extra_info),
        "budget": _default_compute_score_budget(data_source, solution_str, ground_truth, extra_info),
    }

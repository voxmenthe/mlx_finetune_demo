All results are from the mmlu_pro_computer_science task.
mlx_lm.evaluate --model <model> --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

============================================================

Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-8
{
    "alias": "computer_science",
    "exact_match,custom-extract": 0.7926829268292683,
    "exact_match_stderr,custom-extract": 0.020044980247224453
}


Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr3e-7Results:
{
    "alias": "computer_science",
    "exact_match,custom-extract": 0.7926829268292683,
    "exact_match_stderr,custom-extract": 0.020044980247224457
}

Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-8
{
    "alias": "computer_science",
    "exact_match,custom-extract": 0.7878048780487805,
    "exact_match_stderr,custom-extract": 0.02021693788475414
}

Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr1e-6
--data-path voxmenthe/merged-sft-coding-mix2
{
    "alias": "computer_science",
    "exact_match,custom-extract": 0.6219512195121951,
    "exact_match_stderr,custom-extract": 0.023976756269796867
}

Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr2e7
--data-path voxmenthe/merged-sft-coding-mix2
Results:
{
    "alias": "computer_science",
    "exact_match,custom-extract": 0.7292682926829268,
    "exact_match_stderr,custom-extract": 0.02197108846947813
}

Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr4e7
--data-path voxmenthe/merged-sft-coding-mix2
Results:
{
    "alias": "computer_science",
    "exact_match,custom-extract": 0.697560975609756,
    "exact_match_stderr,custom-extract": 0.022711632302604486
}
Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8
Results:
{
    "alias": "computer_science",
    "exact_match,custom-extract": 0.7048780487804878,
    "exact_match_stderr,custom-extract": 0.022552572925167262
}
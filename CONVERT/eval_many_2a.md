echo "Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-7"
mlx_lm.evaluate --model Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-7 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

echo "Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr1e-8"
mlx_lm.evaluate --model Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr1e-8 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

echo "Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-9"
mlx_lm.evaluate --model Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-9 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template
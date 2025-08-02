echo "mlx-community/Qwen3-Coder-30B-A3B-Instruct-bf16"
mlx_lm.evaluate --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-bf16 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

echo "mlx-community/Qwen3-30B-A3B-Thinking-2507-bf16"
mlx_lm.evaluate --model mlx-community/Qwen3-30B-A3B-Thinking-2507-bf16 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template
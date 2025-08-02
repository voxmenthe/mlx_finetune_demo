

echo "mlx-community/XBai-o4-8bit"
mlx_lm.evaluate --model mlx-community/mlx-community/XBai-o4-8bit --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template


echo "mlx-community/XBai-o4-4bit-DWQ"
mlx_lm.evaluate --model mlx-community/XBai-o4-4bit-DWQ --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template
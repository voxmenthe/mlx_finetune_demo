echo "mlx-community/cogito-v2-preview-llama-70B-4Bit"
mlx_lm.evaluate --model mlx-community/cogito-v2-preview-llama-70B-4Bit --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

echo "mlx-community/GLM-4.5-Air-5bit"
mlx_lm.evaluate --model mlx-community/GLM-4.5-Air-5bit --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

https://github.com/cs2764/mlx-quantization
# dynamic quantization


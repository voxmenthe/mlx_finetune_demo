mlx_lm.dwq --model Qwen/Qwen3-Coder-30B-A3B-Instruct --mlx-path Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr3e-7 --max-seq-length 2048 --batch-size 4 --learning-rate 3e-7 --group-size 32 --bits 6
touch Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr3e-7/README.md
mlx_lm.upload --path ./Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr3e-7 --upload-repo mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr3e-7

mlx_lm.dwq --model Qwen/Qwen3-Coder-30B-A3B-Instruct --mlx-path Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr9e-8 --max-seq-length 2048 --batch-size 4 --learning-rate 9e-8 --group-size 32 --bits 6
touch Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr9e-8/README.md
mlx_lm.upload --path ./Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr9e-8 --upload-repo mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit-DWQ-lr9e-8

mlx_lm.evaluate --model Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr3e-7 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template
mlx_lm.evaluate --model Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr9e-8 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

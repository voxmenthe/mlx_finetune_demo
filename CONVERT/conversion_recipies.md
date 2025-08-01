mlx_lm.dwq --model Qwen/Qwen3-30B-A3B-Instruct-2507 --mlx-path Qwen3-30B-A3B-Instruct-2507-4bit-DWQ --max-seq-length 2048 --batch-size 4 --learning-rate 1e-7 --group-size 32 --bits 4
mlx_lm.upload --path ./Qwen3-30B-A3B-Instruct-2507-4bit-DWQ --upload-repo mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ
mlx_lm.generate --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ --max-tokens 4096 --temp 0.7 -p "Explain why the Soviet Union didn't collapse earlier than it did"

mlx_lm.dwq --model Qwen/Qwen3-30B-A3B-Instruct-2507 --mlx-path Qwen3-30B-A3B-Instruct-2507-8bit-DWQ --max-seq-length 2048 --batch-size 4 --learning-rate 8e-8 --group-size 32 --bits 8
Qwen3-30B-A3B-Instruct-2507-8bit-DWQ/README.md
mlx_lm.upload --path ./Qwen3-30B-A3B-Instruct-2507-8bit-DWQ --upload-repo mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit-DWQ

mlx_lm.dwq --model Qwen/Qwen3-30B-A3B-Instruct-2507 --mlx-path Qwen3-30B-A3B-Instruct-2507-6bit-DWQ --max-seq-length 2048 --batch-size 4 --learning-rate 1e-7 --group-size 32 --bits 6
Qwen3-30B-A3B-Instruct-2507-6bit-DWQ/README.md
mlx_lm.upload --path ./Qwen3-30B-A3B-Instruct-2507-6bit-DWQ --upload-repo mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit-DWQ
TODO:
mlx_lm.generate --model mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit-DWQ --max-tokens 4096 --temp 0.7 -p "Explain why the Soviet Union didn't collapse earlier than it did"


"mlabonne/open-perfectblend"

ValueError: Unsupported data format, check the supported formats here:
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#Data.
 --data-path 
        "--data-path",
        type=str,
        default="allenai/tulu-3-sft-mixture",

models: zai-org/GLM-4.5-Air

mlx_lm.dwq --model zai-org/GLM-4.5-Air --mlx-path mlx-community/GLM-4.5-Air-8bit-DWQ --max-seq-length 2048 --batch-size 4 --learning-rate 8e-8 --group-size 32 --bits 8


mlx_lm.dwq --model Qwen/Qwen3-30B-A3B-Instruct-2507 --mlx-path Qwen3-30B-A3B-Instruct-2507-6bit-DWQ --max-seq-length 2048 --batch-size 4 --learning-rate 1e-7 --group-size 32 --bits 6

======== evals
mlx_lm.evaluate --model Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-8 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

mlx_lm.evaluate --model Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr3e-7 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template

mlx_lm.evaluate --model Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-8 --tasks mmlu_pro_computer_science --max-tokens 5000 --no-apply-chat-template
mlx_lm.dwq --model Qwen/Qwen3-Coder-30B-A3B-Instruct --mlx-path Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr2e7 --max-seq-length 2048 --batch-size 4 --learning-rate 2e-7 --group-size 32 --bits 8 --data-path voxmenthe/merged-sft-coding-mix2
touch Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr2e7/README.md
mlx_lm.upload --path ./Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr2e7 --upload-repo mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr2e7

mlx_lm.dwq --model Qwen/Qwen3-Coder-30B-A3B-Instruct --mlx-path Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr4e7 --max-seq-length 2048 --batch-size 4 --learning-rate 4e-7 --group-size 32 --bits 8 --data-path voxmenthe/merged-sft-coding-mix2
touch Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr4e7/README.md
mlx_lm.upload --path ./Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr4e7 --upload-repo mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr4e7

mlx_lm.dwq --model Qwen/Qwen3-Coder-30B-A3B-Instruct --mlx-path Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr9e8 --max-seq-length 2048 --batch-size 4 --learning-rate 9e-8 --group-size 32 --bits 8 --data-path voxmenthe/merged-sft-coding-mix2
touch Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr9e8/README.md
mlx_lm.upload --path ./Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr9e8 --upload-repo mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr9e8

mlx_lm.dwq --model Qwen/Qwen3-Coder-30B-A3B-Instruct --mlx-path Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8 --max-seq-length 2048 --batch-size 4 --learning-rate 5e-8 --group-size 32 --bits 8 --data-path voxmenthe/merged-sft-coding-mix2
touch Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8/README.md
mlx_lm.upload --path ./Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8 --upload-repo mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr5e-8

mlx_lm.dwq --model Qwen/Qwen3-Coder-30B-A3B-Instruct --mlx-path Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr1e-6 --max-seq-length 2048 --batch-size 4 --learning-rate 1e-6 --group-size 32 --bits 8 --data-path voxmenthe/merged-sft-coding-mix2
touch Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr1e-6/README.md
mlx_lm.upload --path ./Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr1e-6 --upload-repo mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit-DWQ-lr1e-6
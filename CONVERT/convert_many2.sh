mlx_lm.dwq --model Qwen/Qwen3-30B-A3B-Instruct-2507 --mlx-path Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-7 --max-seq-length 2048 --batch-size 4 --learning-rate 8e-7 --group-size 32 --bits 6
touch Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-7/README.md
mlx_lm.upload --path ./Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-7 --upload-repo mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr8e-7

mlx_lm.dwq --model Qwen/Qwen3-30B-A3B-Instruct-2507 --mlx-path Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr1e-8 --max-seq-length 2048 --batch-size 4 --learning-rate 1e-8 --group-size 32 --bits 6
touch Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr1e-8/README.md
mlx_lm.upload --path ./Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr1e-8  --upload-repo mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr1e-8

mlx_lm.dwq --model Qwen/Qwen3-30B-A3B-Instruct-2507 --mlx-path Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-9 --max-seq-length 2048 --batch-size 4 --learning-rate 5e-9 --group-size 32 --bits 6
touch Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-9/README.md
mlx_lm.upload --path ./Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-9 --upload-repo mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit-DWQ-lr5e-9

#========================================

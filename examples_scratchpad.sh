

mlx_lm.lora \
    --model mlx_models/Qwen3-4B-mlx \
    --adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \
    --data DATA/SACREDHUNGER \
    --test


cat temp_prompt.txt | python src/inference/generate_qwen3.py \
  --model-path mlx_models/Qwen3-4B-mlx \
  --adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \
  --prompt "-"

cat temp_prompt.txt | python src/inference/generate_qwen3.py \
--model-path mlx_models/Qwen3-4B-mlx \
--adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \
--prompt "-" \
--repetition-penalty 1.1 \
--temp 0.75 \
--top-p 0.95 

# WITHOUT ADAPTER
cat temp_prompt.txt | python src/inference/generate_qwen3.py \
--model-path mlx_models/Qwen3-4B-mlx \
--prompt "-" \
--repetition-penalty 1.1 \
--temp 0.75 \
--top-p 0.95 

python prepare_training_data.py \
    --input_files semantic_chunks_480.json semantic_chunks_520.json semantic_chunks_680.json semantic_chunks_790.json \
    --output_dir ../../DATA/SACREDHUNGER/ \
    --train_ratio 0.9 \
    --seed 123

python prepare_training_data.py \
    --input_files sacredhunger_350.json sacredhunger_480.json sacredhunger_520.json sacredhunger_570.json sacredhunger_680.json sacredhunger_730.json sacredhunger_790.json \
    --output_dir ../../DATA/SACREDHUNGER/ \
    --train_ratio 0.93 \
    --seed 211

python prepare_training_data.py \
    --input_files allthekingsmen_480.json allthekingsmen_520.json allthekingsmen_680.json allthekingsmen_790.json \
    --output_dir ../../DATA/ALLTHEKINGSMEN/ \
    --train_ratio 0.9 \
    --seed 123

cat temp_prompt.txt | python src/inference/generate_qwen3.py \
--model-path mlx_models/Qwen3-14B-mlx \
--adapter-path ADAPTERS/qwen3_14b_lora_sacredhunger_multi \
--prompt "-" \
--repetition-penalty 1.1 \
--temp 0.75 \
--top-p 0.95 

cat temp_prompt.txt | python src/inference/generate_qwen3.py \
--model-path mlx_models/Qwen3-14B-mlx \
--adapter-path ADAPTERS/qwen3_14b_dora_novels_sh_atkm \
--prompt "-" \
--repetition-penalty 1.1 \
--temp 0.75 \
--top-p 0.95 

books:
Fatherland_HarrisRobert.txt
Great Gatsby, The - Francis Scott Fitzgerald.txt
Imperium_ANovelofAncientRo_HarrisRobert.txt
LOTR.txt
One Hundred Years of Solitude - Gabriel Garcia Marquez.txt
OldManandtheSeaThe_ErnestHemingway.txt
Pride_and_Prejudice.txt
PaperTowns_JohnGreen.txt
Pachinko_MinJinLee.txt
RedSister-MarkLawrence.txt
ToKillAMockingbird_HarperLee.txt
TheMartian.txt
TheMagicians1.txt
TheMagicians2.txt
TheMagicians3.txt
TheLiontheWitchandtheWar_LewisCS_.txt
TheGodfather.txt
TheGraveyardBook.txt
TheDaVinciCode_BrownDan.txt
AWrinkleinTime(PuffinModer_LengleMadeleine.txt
AdventuresofTomSawyerThe_MarkTwain.txt
Bartimaeus1.txt
Bartimaeus2.txt
Bartimaeus3.txt
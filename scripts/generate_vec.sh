gpu=${1:-0}


CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait sycophantic \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name sycophantic \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait sycophantic \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name non-sycophantic \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/sycophantic_neg_instruct.csv \
    --trait sycophantic \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/ \
    --threshold 50

# Contrarian trait
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait contrarian \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/contrarian_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name contrarian \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait contrarian \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/contrarian_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/contrarian_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/contrarian_neg_instruct.csv \
    --trait contrarian \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/ \
    --threshold 50

# Highly disagreeable trait
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait highly_disagreeable \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/highly_disagreeable_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name disagreeable \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait highly_disagreeable \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/highly_disagreeable_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name agreeable \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/highly_disagreeable_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/highly_disagreeable_neg_instruct.csv \
    --trait highly_disagreeable \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/ \
    --threshold 50

# APE trait
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait APE \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/APE_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name persuasive \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait APE \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/APE_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name balanced \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/APE_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/APE_neg_instruct.csv \
    --trait APE \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/ \
    --threshold 50
#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# 声明关联数组
declare -A MODEL_PATH
MODEL_PATH["llava-1.5"]="/mnt-nfsdata/model_base/llava-1.5-7b-hf"
MODEL_PATH["llava-next"]="/mnt-nfsdata/model_base/llava-v1.6-vicuna-7b-hf"
MODEL_PATH["instructblip"]="/mnt-nfsdata/model_base/instructblip-vicuna-7b"


############################################POPE Evaluation Script########################################

# # 参数设置
DAMRO_ALPHA=0.5
DAMRO_TOPK=4
DAMRO_BETA=0.1
MODEL="instructblip"   # "llava-1.5" | "llava-next" | "instructblip"
TYPE="random"        # "random" | "adversarial"

# 路径变量引用时要加引号！
python /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/eval_pope/run_pope.py \
   --model_path "${MODEL_PATH[$MODEL]}" \
   --image_folder /mnt-nfsdata/gongxuan/data-1/gx_damro/data/coco/val2014 \
   --question_file /mnt-nfsdata/gongxuan/data-1/gx_damro/data/pope/coco/coco_pope_${TYPE}.jsonl \
   --answers_file /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_pope_${MODEL}_${TYPE}_alpha${DAMRO_ALPHA}_beta${DAMRO_BETA}_topk${DAMRO_TOPK}_direct2.jsonl \
   --damro_alpha ${DAMRO_ALPHA} \
   --damro_topk ${DAMRO_TOPK} \
   --damro_beta ${DAMRO_BETA} \
   --use_damro \
   --seed 42 \
   --batch_size 1

############################################CHAIR Evaluation Script########################################


#参数设置
DAMRO_ALPHA=1.5
DAMRO_TOPK=4
DAMRO_BETA=0.1
MODEL="instructblip"   # "llava-1.5" | "llava-next" | "instructblip"
TYPE="random"        # "random" | "adversarial"

# 路径变量引用时要加引号！
python /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/eval_chair/run_chair.py \
   --model_path "${MODEL_PATH[$MODEL]}" \
   --image_folder /mnt-nfsdata/gongxuan/data-1/gx_damro/data/coco/val2014 \
   --question_file /mnt-nfsdata/gongxuan/data-1/gx_damro/data/chair/coco_pope_${TYPE}.jsonl \
   --answers_file /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_chair_${MODEL}_${TYPE}_alpha${DAMRO_ALPHA}_beta${DAMRO_BETA}_topk${DAMRO_TOPK}_direct2.jsonl \
   --damro_alpha ${DAMRO_ALPHA} \
   --damro_topk ${DAMRO_TOPK} \
   --damro_beta ${DAMRO_BETA} \
   --use_damro \
   --seed 42 \
   --batch_size 1

############################################ MME Evaluation Script########################################



DAMRO_ALPHA=2
DAMRO_TOPK=10
DAMRO_BETA=0.1
MODEL="llava-1.5"   # "llava-1.5" | "llava-next" | "instructblip"
TYPE="random"        # "random" | "adversarial"

# 路径变量引用时要加引号！
python /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/eval_mme/run_mme.py \
   --model_path "${MODEL_PATH[$MODEL]}" \
   --question_file /mnt-nfsdata/gongxuan/data-1/gx_damro/data/mme \
   --answers_file /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_mme_${MODEL}_${TYPE}_alpha${DAMRO_ALPHA}_beta${DAMRO_BETA}_topk${DAMRO_TOPK}.jsonl \
   --damro_alpha ${DAMRO_ALPHA} \
   --damro_topk ${DAMRO_TOPK} \
   --damro_beta ${DAMRO_BETA} \
   --use_damro \
   --seed 42 \
   --batch_size 1



############################################ GPT4V Evaluation Script########################################



DAMRO_ALPHA=2
DAMRO_TOPK=10
DAMRO_BETA=0.1
MODEL="llava-1.5"   # "llava-1.5" | "llava-next" | "instructblip"
TYPE="random"        # "random" | "adversarial"

# 路径变量引用时要加引号！
python    /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/gpt4v_aided/run_gpt4v_aided.py \
   --model_path "${MODEL_PATH[$MODEL]}" \
   --image_folder /mnt-nfsdata/gongxuan/data-1/gx_damro/data/coco/val2014 \
   --question_file /mnt-nfsdata/gongxuan/data-1/gx_damro/data/gpt4v/gpt4v.jsonl \
   --answers_file /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_gpt4v_${MODEL}_${TYPE}_alpha${DAMRO_ALPHA}_beta${DAMRO_BETA}_topk${DAMRO_TOPK}.jsonl \
   --damro_alpha ${DAMRO_ALPHA} \
   --damro_topk ${DAMRO_TOPK} \
   --damro_beta ${DAMRO_BETA} \
   --use_damro \
   --seed 42 \
   --batch_size 1


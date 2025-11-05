############################################POPE Evaluation Script########################################
TYPE="random"  #or "random","adversarial"

python /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/eval_pope/eval_pope.py \
   --gt_files /mnt-nfsdata/gongxuan/data-1/gx_damro/data/pope/coco/coco_pope_${TYPE}.jsonl \
   --gen_files /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_pope_instructblip_random_alpha0.5_beta0.1_topk4_direct2.jsonl

############################################CHAIR Evaluation Script########################################

python /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/eval_chair/eval_chair.py \
   --cap_file /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_chair_instructblip_random_alpha1.5_beta0.1_topk4_direct2.jsonl \
   --image_id_key "image_id" \
   --caption_key "caption" \
   --cache /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/eval_chair/chair.pkl \
   --coco_path /mnt-nfsdata/gongxuan/data-1/gx_damro/data/coco/annotations \
   --save_path /mnt-nfsdata/gongxuan/data-1/gx_damro/output/chair_detail/damro_eval_chair_instructblip_random_alpha1.5_beta0.1_topk4_detail.json

############################################MME Evaluation Script########################################


python /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/eval_mme/eval_mme.py \
--gen_file /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_mme_llava-next_random_alpha2_beta0.1_topk10.jsonl

############################################ GPT4V Evaluation Script########################################


python /mnt-nfsdata/gongxuan/data-1/gx_damro/evaluation/gpt4v_aided/eval_gpt4v.py \
--file_path1 /mnt-nfsdata/gongxuan/data-1/gx_damro/output/original_eval_gpt4v_llava-1.5.jsonl \
--file_path2 /mnt-nfsdata/gongxuan/data-1/gx_damro/output/damro_eval_gpt4v_llava-1.5_alpha2_beta0.1_topk10.jsonl \
--save_path /mnt-nfsdata/gongxuan/data-1/gx_damro/output/gpt4v_detail/damro_eval_gpt4v_llava-1.5_alpha2_beta0.1_topk10_detail.json \
--image_folder /mnt-nfsdata/gongxuan/data-1/gx_damro/data/coco/val2014
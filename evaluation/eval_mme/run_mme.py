import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from transformers import set_seed

from   sample_method.sample import modify_sampling
modify_sampling()            #replace the sample function in transformer lib
from transformers import AutoModelForVision2Seq, AutoProcessor
from utils.conversation import conv_templates

from datasets import load_dataset



def eval_model(args):
    model_path = os.path.expanduser(args.model_path)

    processor = AutoProcessor.from_pretrained(model_path,use_fast=False)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cuda"
    )

    dataset = load_dataset(args.question_file)

    questions = [q for q in dataset['test']]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    batch_size = args.batch_size

    with open(answers_file, "w") as ans_file:
        for i in tqdm(range(0, len(questions), batch_size)):
            batch = questions[i : i + batch_size]

            images = [b["image"] for b in batch]
            prompts = [ b["question"] for b in batch]
            
            if "llava" in model.config.model_type.lower():
                prompts = ["<image>\n" + b["question"]  for b in batch]
                conv_prompts = []
                for qs in prompts:
                    conv = conv_templates["llava_v1"].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    conv_prompts.append(conv.get_prompt())
                prompts = conv_prompts
            
            # print(f'prompts: {prompts}')   #--- IGNORE ---
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(model.device, torch.float32)
         
            if "instructblip" in model.config.model_type.lower():
                if inputs['pixel_values'].dim() == 5:  # [N, B, C, H, W]
                    inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
                max_new_tokens = None
                max_length = 1024
                min_length=1
            else:
                max_new_tokens=1024
                max_length=None
                min_length=None
                

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    use_damro=args.use_damro,
                    damro_alpha=args.damro_alpha,
                    damro_topk=args.damro_topk,
                    damro_beta=args.damro_beta,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    min_length=min_length,
                    use_cache=True,
                )
            
            if "instructblip" in model.config.model_type.lower():
                outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                outputs=[outputs]
            else:
                outputs = processor.batch_decode(
                    generated_ids[:, inputs['input_ids'].shape[-1]:],
                    skip_special_tokens=True
                )

            for b, output in zip(batch, outputs):
                ans_file.write(json.dumps({
                    "question_id": b["question_id"],
                    "gt_answer": b["answer"],
                    "category": b["category"],
                    "prompt": b["question"],
                    "output": output,
                    "model_id": args.model_path,
                }) + "\n")
            ans_file.flush()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--use_damro", action="store_true", help="Enable DAMRO sampling")
    parser.add_argument("--damro_alpha", type=float, default=2)
    parser.add_argument("--damro_topk", type=int, default=10)
    parser.add_argument("--damro_beta", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    
  
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"input:{args.question_file}")
    print(f'seed:{args.seed}')
    print(f'output:{args.answers_file}')
    print(f'batch_size:{args.batch_size}')
    print(f'use_damro:{args.use_damro}')


  
    

    eval_model(args)
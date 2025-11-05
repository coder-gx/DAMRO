import json

from calculation import calculate_metrics
import argparse

import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_mme(ans_file_path):
    data_types={}
    with open(ans_file_path, "r") as f:
        answers = [json.loads(line) for line in f.readlines()]

        for ans in answers:
            if ans['category'] not in data_types:
                data_types[ans['category']] = [ans]
            else:
                data_types[ans['category']].append(ans)
    tmp_dir="./evaluation/eval_mme/Your_Results"
    os.makedirs(tmp_dir, exist_ok=True)

    for data_type in data_types:
        with open(f"{tmp_dir}/{data_type}.txt", "w") as f:
            for ans in data_types[data_type]:
                image_name = ans['question_id'].split('/')[-1]
                f.write(image_name + "\t" + ans['prompt'] + "\t" + ans['gt_answer'] + "\t" + ans['output'].replace("\n", " ") + "\n")
    
    cal = calculate_metrics()
    cal.process_result(tmp_dir)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_file', default='/mnt-nfsdata/gongxuan/data-1/gx_damro/output/test.jsonl', type=str)

    args = parser.parse_args()
    gen_file = args.gen_file

    evaluate_mme(gen_file)

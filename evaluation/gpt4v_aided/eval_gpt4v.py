import base64
import re
from openai import OpenAI
import json
from tqdm import tqdm
import argparse

GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinationsshould be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not countas necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''





def call_api(prompt, image_path):

    client = OpenAI()

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the Base64 string
    base64_image = encode_image(image_path)


    response = client.responses.create(
        model="gpt-4o-2024-05-13",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": prompt },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    return  response


def get_gpt4v_answer(prompt, image_path):
    for _ in range(5):
        try:
            res = call_api(prompt, image_path)
            
            if  res is not None:
                return res.choices[0].message.content
            else:
                print("No response received, retrying...")
        except Exception as e:
            print("Bad request error and retry:", e)
            pass
    print("Failed to get response from GPT-4V after several attempts.")



def main(args):

    tests1= [json.loads(q) for q in open(args.file_path1, "r")]
    tests2= [json.loads(q) for q in open(args.file_path2, "r")]

    avg_hal_score_1 = 0
    avg_hal_score_2 = 0
    avg_det_score_1 = 0
    avg_det_score_2 = 0
    num_count = 0


    with open(args.save_path, "w") as f_save:
        assert len(tests1) == len(tests2), "The number of answers in two files should be the same."

        for i in tqdm(range(len(tests2))):
            test1=tests1[i]
            test2=tests2[i]
            
            assert test1[i]['image'] == test2[i]['image'], "The image names in two files should be the same."
            assert test1[i]['id'] == test2[i]['id'], "The image names in two files should be the same."
            assert test1[i]['prompt'] == test2[i]['prompt'], "The image names in two files should be the same."
            
            
            image_path = args.image_folder + "/"+ tests1[i]["image"]
            id=test1['id']
            qu = test1["prompt"]
            model_response_1 = test1['output']
            model_response_2 = test2['output']
            
            ok=False
            for _ in range(5):

                # gpt-4v eval
                prompt = GPT_JUDGE_PROMPT.format(model_response_1, model_response_2)

                gpt_answer = get_gpt4v_answer(prompt, image_path)

                split_chars = r"[\n*]+"
                ans=re.split(split_chars,gpt_answer.split("Accuracy: ")[-1])
                a=ans[0].split(" ")
            
                
                bns=re.split(split_chars,gpt_answer.split("Detailedness: ")[-1])
                b=bns[0].split(" ")
            
                try:
                    hal_score_1, hal_score_2 = a
                    det_score_1, det_score_2 =b
                    ok = True
                except:
                    print("GPT-4V response parsing error, retrying...")
                    continue #重新生成请求
                
                if not ok:
                    print("Failed to parse GPT-4V response after 5 attempts.")
                    hal_score_1, hal_score_2 = 0,0
                    det_score_1, det_score_2 =0,0

                
    
        avg_hal_score_1 += int(hal_score_1)
        avg_hal_score_2 += int(hal_score_2)
        avg_det_score_1 += int(det_score_1)
        avg_det_score_2 += int(det_score_2)
        num_count += 1
       

        f_save.write(json.dumps({"image_id": id,
                                    "question": qu,
                                    "ans1": model_response_1,
                                    "ans2": model_response_2,
                                    "gpt": gpt_answer,
                                        }) + "\n")
        f_save.flush()

    print("=========================================")
    avg_score_1 = float(avg_hal_score_1) / num_count
    avg_score_2 = float(avg_hal_score_2) / num_count
    avg_score_3 = float(avg_det_score_1) / num_count
    avg_score_4 = float(avg_det_score_2) / num_count
    print(f"The avg hal score for Assistant 1 and Assistent 2: {avg_score_1}; {avg_score_2}")
    print(f"The avg det score for Assistant 1 and Assistent 2: {avg_score_3}; {avg_score_4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path1', type=str)
    parser.add_argument('--file_path2', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--save_path', type=str)
    

    args = parser.parse_args()

    main(args)
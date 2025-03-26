import os
from PIL import Image
import numpy as np
import torch
import re
import argparse

from radvlm.data.utils import  process_sbb
from radvlm.data.datasets import  MIMIC_Dataset_MM
from radvlm.evaluation.models_loading_inference import load_model_and_processor, inference_radialog, inference_llavaov, inference_llavamed
from radvlm.data.utils import process_sbb, inference_gpt4o_with_retry, setup_azure_openai
from radvlm import DATA_DIR


parser = argparse.ArgumentParser(description="A script to evaluate conversations with GPT-4o.")
parser.add_argument("--grounding", action="store_true",
                    help="Set this flag to evaluate grounded conversations")
parser.add_argument('--model_name', type=str, default='radialog', help="The VLM to evaluate")
parser.add_argument("--azure_model", type=str, required=True,
                        help="The azume model name (gpt-4o, gpt-4o-mini, etc.) used to generate conversations ")
args = parser.parse_args()


script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(script_dir, "results")

split = "test"
datasetpath = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
gender_file = os.path.join(datasetpath, 'genders.json')
filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports')

if args.grounding:
    sentencesBBoxpath = os.path.join(DATA_DIR, 'MS-CXR/sentences_and_BBox_mscxr')
    conversation_dir= os.path.join(datasetpath, 'conversations/test/grounding')
else:
    sentencesBBoxpath = None
    conversation_dir= os.path.join(datasetpath, 'conversations/test/standard')

input_dataset = MIMIC_Dataset_MM(
    datasetpath = datasetpath,
    split="test", flag_img=True, 
    flag_lab=True, only_frontal=True, 
    flag_instr=True, 
    filtered_reports_dir=filtered_reports_dir,
    sentencesBBoxpath = sentencesBBoxpath,
    conversation_dir=conversation_dir,
    genderpath=gender_file,
    classif=False,
    seed=0)


print(len(input_dataset))

prefix_file_path = os.path.join(script_dir, 'prefix_evaluate_conv.txt')
with open(prefix_file_path, 'r') as file:
        prefix_content = file.read()

tokenizer, model, processor = load_model_and_processor(args.model_name, device_map='auto')

# Initialize a list to store scores
scores = []

for i in range(len(input_dataset)):
    imgpath = input_dataset[i]['img_path']
    image_id = os.path.splitext(os.path.basename(imgpath))[0]

    report = input_dataset[i]['txt']
    image = input_dataset[i]['img']
    # bounding_boxes = input_dataset[i]['boxes']
    sentencesBBox = input_dataset[i]['sentencesBBox']
    labels = input_dataset[i]['labels']
    report = input_dataset[i]['txt']
    view = input_dataset[i]['view']
    gender = input_dataset[i]['gender']
    if gender is not None:
        gender = 'female' if gender == 'F' else 'male'
    gt_conversation = input_dataset[i]['conversation']
    
    prompt = prefix_content + "Radiology report: " + report + "\n"
    prompt = prompt + "List of Abnormalities: " + ", ".join(labels) + "\n"
    prompt = prompt + "View: " + str(view) + "\n"
    prompt += "Gender: " + str(gender) + "\n"
    if sentencesBBox is not None:
        processed_sbb = process_sbb(sentencesBBox)
        if processed_sbb is not None:
            prompt = prompt + "Selected observations with bounding boxes coordinates:\n" + processed_sbb + "\n"
    prompt = prompt + "Here is the conversation to evaluate: " + "\n\n"
    
    chat_history = [] 
    try:
        for j in range(len(gt_conversation)):
            if gt_conversation[j]["from"] == "human":
                question = gt_conversation[j]["value"]
                prompt = prompt + "User: " + question + "\n" 
            else:
                expected_answer = gt_conversation[j]["value"]
                prompt = prompt + "Expected answer: " + expected_answer + "\n"

                # Generate response from the model 
                image = Image.open(imgpath)
                with torch.no_grad():
                    if args.model_name == 'radialog':
                        response, chat_history = inference_radialog(tokenizer, model, imgpath, question, chat_history)
                    elif args.model_name == 'llavamed':
                        response, chat_history = inference_llavamed(model, processor, imgpath, question, chat_history)
                    else:
                        response, chat_history = inference_llavaov(model, processor, imgpath, question, chat_history)
                prompt = prompt + "Generated answer: " + response + "\n\n"

    except Exception as e:
        # Log the error and skip the current item in the main loop
        print(f"Error during inference at dataset index {i}: {e}")
        continue
    
    prompt = prompt + "Note: write the overall score (/10) this way, so I can extract it: Overall score: <score>/10" + "\n"

    client = setup_azure_openai()

    generated_text = inference_gpt4o_with_retry(prompt, client, args.azure_model)

    print("-------------------------------------\n\n\n\n")

    print(prompt)
    print(imgpath)
    print(generated_text)
    pattern = r'(?i)overall\s*score.*?([\d\.]+)/10'

    # Use DOTALL so ".*?" can span multiple lines if needed
    matches = re.findall(pattern, generated_text, flags=re.DOTALL)


    if matches:
        for match in matches:
            try:
                numeric_score = float(match)
                # Ensure the numeric score is <= 10
                if numeric_score <= 10:
                    scores.append(numeric_score)
                else:
                    pass
            except ValueError:
                print(f"Non-numeric score encountered: '{match}'")
    else:
        print("No valid 'Overall score: X/10' found in the generated text.")


    model_name = os.path.basename(args.model_name)

    average_score = np.mean(scores)
    print(scores)
    print(f"RUNNING AVERAGE SCORE: {average_score}")

    if args.grounding:
        average_score_file = os.path.join(OUTPUT_DIR, f"average_score_grounding_{model_name}.txt")
    else:
        average_score_file = os.path.join(OUTPUT_DIR, f"average_score_{model_name}.txt")

    with open(average_score_file, "w") as f:
        f.write(str(average_score))


print("EVALUATION COMPLETED")


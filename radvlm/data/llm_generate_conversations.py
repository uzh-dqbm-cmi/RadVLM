import argparse
import json
import os
import torch
from multiprocessing import Pool

from radvlm.data.datasets import *
from radvlm.data.utils import process_sbb, inference_gpt4o_with_retry, setup_azure_openai
from radvlm import DATA_DIR


def create_conversation_dataset(input_dataset, prefix_file_path, output_dir, client, azure_model):
    # Read the prompt prefix from file
    with open(prefix_file_path, 'r') as file:
        prefix_content = file.read()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(input_dataset)):
        # Stop if the output directory already has 100,000 files
        if len(os.listdir(output_dir)) >= 100000:
            print("Reached 100,000 files. Interrupting the loop.")
            break

        imgpath = input_dataset[i]['img_path']
        image_id = os.path.splitext(os.path.basename(imgpath))[0]
        output_file_path = os.path.join(output_dir, f'{image_id}.json')
        if os.path.exists(output_file_path):
            print("Already done, skip!")
            continue

        report = input_dataset[i]['txt']
        if report=='None':
            print("Report is None")
            continue

        sentencesBBox = input_dataset[i].get('sentencesBBox', None)
        view = input_dataset[i].get('view', None)
        gender = input_dataset[i].get('gender', None)
        if gender is not None:
            gender = 'female' if gender == 'F' else 'male'
        labels = input_dataset[i]['labels']

        # Build the prompt for GPT4o inference
        prompt = prefix_content + "Radiology report: " + report + "\n"
        prompt += "List of Abnormalities: " + ", ".join(labels) + "\n"
        prompt += "View: " + str(view) + "\n"
        prompt += "Gender: " + str(gender) + "\n"
        if sentencesBBox and process_sbb(sentencesBBox):
            prompt += "Selected observations with bounding boxes coordinates:\n" + process_sbb(sentencesBBox) + "\n"

        prompt += "\nConversation in expected format:\n"
        print(prompt)

        generated_text = inference_gpt4o_with_retry(prompt, client, azure_model)
        print(generated_text)
        print("--------------------------------------")

        # Try to extract JSON content from the generated text
        try:
            start_idx = generated_text.index("[")
            end_idx = generated_text.rindex("]") + 1
            extracted_content = generated_text[start_idx:end_idx]
            extracted_list = json.loads(extracted_content)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Could not extract a valid JSON list: {e}")
            extracted_list = None

        if isinstance(extracted_list, list):
            with open(output_file_path, 'w') as json_file:
                json.dump(extracted_list, json_file, indent=4)
                print("Output saved!")
        else:
            print("Could not extract a list")


def process_chunk(chunk_index, chunk, prefix_file_path, output_dir, azure_model):
    print(f"Processing chunk {chunk_index} on process {os.getpid()}")
    # Initialize the Azure OpenAI client within the child process
    client = setup_azure_openai()
    create_conversation_dataset(chunk, prefix_file_path, output_dir, client, azure_model)


def main():
    parser = argparse.ArgumentParser(
        description="Conversation dataset creation script with GPT4o inference."
    )
    parser.add_argument("--azure_model", type=str, required=True,
                        help="The azure model name (gpt-4o, gpt-4o-mini, etc.) used to generate conversations")
    parser.add_argument("--split", choices=['train', 'test'], type=str, required=True,
                        help="The dataset split")
    parser.add_argument("--grounding", action="store_true",
                        help="Set this flag to generate grounded conversations.")
    parser.add_argument("--padchest", action="store_true",
                        help="Set this flag to generate conversations for padchest dataset.")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="Number of parallel chunks.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prepare the dataset and prompt based on the provided flags
    if not args.padchest:
        datasetpath = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
        filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG', 'filtered_reports')
        datasetpath_mscxr = os.path.join(DATA_DIR, 'MS-CXR')
        gender_file = os.path.join(datasetpath, 'genders.json')
        if args.grounding:
            sentencesBBoxpath = os.path.join(datasetpath_mscxr, 'sentences_and_BBox_mscxr')
            prefix_file_path = os.path.join(script_dir, 'prefixes_prompts/prefix_conv_grounding.txt')
            folder_name = 'grounding'
        else:
            sentencesBBoxpath = None  # using the full MIMIC-CXR dataset
            prefix_file_path = os.path.join(script_dir, 'prefixes_prompts/prefix_conv.txt')
            folder_name = 'standard'

        split = args.split
        dataset = MIMIC_Dataset_MM(
            datasetpath=datasetpath,
            split=split,
            flag_img=False,
            flag_lab=True,
            only_frontal=True,
            flag_instr=False,
            filtered_reports_dir=filtered_reports_dir,
            sentencesBBoxpath=sentencesBBoxpath,
            genderpath=gender_file,
            classif=False,
            seed=0
        )
        print(f"Total dataset size: {len(dataset)}")
    else:
        datasetpath = os.path.join(DATA_DIR, 'PadChest')
        split = args.split
        dataset = PadChest_grounding_per_image(
            datasetpath=datasetpath,
            split=split,
            flag_instr=False
        )
        prefix_file_path = os.path.join(script_dir, 'prefixes_prompts/prefix_conv.txt')
        folder_name = 'padchest'

    output_dir = os.path.join(datasetpath, 'conversations', split, folder_name)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Ensure reproducibility
    torch.manual_seed(125)

    # Split the dataset into chunks for parallel processing
    dataset_size = len(dataset)
    num_chunks = args.num_chunks
    chunk_size = dataset_size // num_chunks
    remainder = dataset_size % num_chunks
    split_sizes = [chunk_size + 1 if i < remainder else chunk_size for i in range(num_chunks)]
    chunks = torch.utils.data.random_split(dataset, split_sizes)

    # Use multiprocessing Pool to process all chunks concurrently
    with Pool(processes=num_chunks) as pool:
        pool.starmap(
            process_chunk,
            [(i, chunks[i], prefix_file_path, output_dir, args.azure_model)
             for i in range(num_chunks)]
        )


if __name__ == "__main__":
    main()

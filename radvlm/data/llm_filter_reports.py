import os
import argparse
import torch
import random
from torch.utils.data import random_split
from multiprocessing import Pool

from radvlm.data.utils import inference_gpt4o_with_retry, setup_azure_openai
from radvlm.data.datasets import MIMIC_Dataset_MM, CheXpertPlus_Dataset
from radvlm import DATA_DIR


def extract_findings_for_chunk(input_chunk, prefix_file_path, output_dir, client, azure_model, chexpertplus):
    """
    Processes a chunk of the dataset, extracting the findings for each sample
    and storing them in the specified output folder using GPT4o for inference.
    """
    # Read the prompt from the text file
    with open(prefix_file_path, 'r') as file:
        prefix_content = file.read()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(input_chunk)):
        print("----------------------------------------------------")
        # Retrieve the image path
        imgpath = input_chunk[i]['img_path']
        print(input_chunk[i]['study_id'])

        # Name the output file by image_id or study_id
        if chexpertplus:
            txt_path = "_".join(imgpath.split('/')[-4:-1]) + ".txt"
            output_file_path = os.path.join(output_dir, txt_path)
        else:
            study_id = input_chunk[i].get("study_id", None)
            if study_id:
                output_file_path = os.path.join(output_dir, f"{study_id}.txt")
            else:
                image_id = os.path.splitext(os.path.basename(imgpath))[0]
                output_file_path = os.path.join(output_dir, f"{image_id}.txt")

        # Skip if already processed
        if os.path.exists(output_file_path):
            print(f"{output_file_path} already exists, skip!")
            continue

        # Get the report text
        report = input_chunk[i]['txt']
        if report is None:
            continue

        # Create the prompt
        prompt = prefix_content + report + "\n    - Extracted findings:\n"


        # Perform inference using GPT4o
        generated_text = inference_gpt4o_with_retry(prompt, client, azure_model)

        print("Generated text:")
        print(generated_text)

        # Save the generated text if it is valid
        if not generated_text or "None" in generated_text:
            print("Empty text or 'None' found; skipping save.")
            continue
        else:
            with open(output_file_path, 'w') as output_file:
                output_file.write(generated_text)


def process_chunk(chunk_index, chunk, prefix_file_path, output_dir, chexpertplus, azure_model):
    print(f"Processing chunk {chunk_index} on process {os.getpid()}")
    client = setup_azure_openai()
    extract_findings_for_chunk(chunk, prefix_file_path, output_dir, client, azure_model, chexpertplus)


def main():
    parser = argparse.ArgumentParser(
        description="Filter reports script with GPT4o inference (parallel processing)."
    )
    parser.add_argument("--azure_model", type=str, required=True,
                        help="The azure model name (gpt-4o, gpt-4o-mini, etc.) used to generate conversations")
    parser.add_argument("--chexpertplus", action="store_true",
                        help="Set this flag to process CheXpertPlus dataset logic (naming by image_id).")
    parser.add_argument("--split", choices=['train', 'test'], type=str, required=True,
                        help="The dataset split")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="How many total chunks to split the dataset into (number of parallel processes).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize the Azure OpenAI client once in the main process
    if args.chexpertplus:
        prefix_file_path = os.path.join(script_dir, 'prefixes_prompts/prefix_filter_reports_cplus.txt')
    else:
        prefix_file_path = os.path.join(script_dir, 'prefixes_prompts/prefix_filter_reports.txt')

    # Example for MIMIC-CXR:
    split = args.split
    if args.chexpertplus:
        # Use CheXpertPlus dataset logic
        datasetpath = os.path.join(DATA_DIR, 'CheXpertPlus')
        output_dir = os.path.join(DATA_DIR, 'CheXpertPlus', 'filtered_reports')
        dataset = CheXpertPlus_Dataset(datasetpath=datasetpath, split=split, only_frontal=True, flag_img=False, filtered_reports_dir=output_dir)

    else:
        # Use MIMIC dataset by default
        datasetpath = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
        output_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG', 'filtered_reports')
        dataset = MIMIC_Dataset_MM(
            datasetpath=datasetpath,
            split=split,
            filtered_reports_dir=output_dir,
            flag_img=False,
            flag_lab=False,
            only_frontal=True
        )


    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    print("Total dataset size:", len(dataset))
    torch.manual_seed(11)
    random.seed(11)

    # Split the dataset into chunks for parallel processing
    dataset_size = len(dataset)
    num_chunks = args.num_chunks
    chunk_size = dataset_size // num_chunks
    remainder = dataset_size % num_chunks
    split_sizes = [chunk_size + 1 if i < remainder else chunk_size for i in range(num_chunks)]
    chunks = random_split(dataset, split_sizes)

    # Use multiprocessing to process all chunks concurrently
    with Pool(processes=num_chunks) as pool:
        pool.starmap(
            process_chunk,
            [(i, chunks[i], prefix_file_path, output_dir, args.chexpertplus, args.azure_model)
             for i in range(num_chunks)]
        )


if __name__ == "__main__":
    main()


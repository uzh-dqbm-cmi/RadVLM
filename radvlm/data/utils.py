import numpy as np
import os
import torch
import torchvision.transforms.v2 as transforms
import pandas as pd
import time

from ensemble_boxes import weighted_boxes_fusion

from openai import AzureOpenAI


def setup_azure_openai():

    api_key = os.environ.get('AZURE_OPENAI_API_KEY')
    if api_key is None:
        raise EnvironmentError("The environment variable 'AZURE_OPENAI_API_KEY' is not set.")

    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
    if endpoint is None:
        raise EnvironmentError("The environment variable 'AZURE_OPENAI_ENDPOINT' is not set.")
    
    api_version = os.environ.get('AZURE_API_VERSION')
    if api_version is None:
        raise EnvironmentError("The environment variable 'AZURE_API_VERSION' is not set.")
    


    client = AzureOpenAI(
            azure_endpoint=endpoint,   # e.g. "https://<your-resource-name>.openai.azure.com"
            api_key=api_key,           # Your Azure OpenAI key
            api_version=api_version                            # Example API version (use the one you have)
        )
    return client

    

def process_sbb(data):
    # Initialize a dictionary to hold the sentences and their bounding boxes
    sentence_dict = {}

    # Process each item in the list
    for item in data:
        sentence = item["observation"]
        bounding_box = []

        if "box" not in item:
            continue 

        # Convert each coordinate to float if possible, and format to the second decimal place
        for coord in item["box"]:
            try:
                # Attempt to convert to float and format
                bounding_box.append(f"{float(coord):.2f}")
            except ValueError:
                # If conversion fails, return None
                return None
        
        bounding_box_str = f"[{', '.join(bounding_box)}]"
        
        # If the sentence is already in the dictionary, append the bounding box
        if sentence in sentence_dict:
            sentence_dict[sentence].append(bounding_box_str)
        else:
            sentence_dict[sentence] = [bounding_box_str]

    # Create the final string
    result_lines = [f"{sentence}: {' '.join(bounding_boxes)}" for sentence, bounding_boxes in sentence_dict.items()]
    result = "\n".join(result_lines)
    
    return result



def inference_gpt4o_with_retry(prompt, client, azure_model, max_retries=20):
    for attempt in range(max_retries):
        try:
            # Use the OpenAI client to make the request
            completion = client.chat.completions.create(
                model=azure_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
            )

            # Extract the response content
            response_text = completion.choices[0].message.content
            # If response_text is None, exit immediately without retrying
            if response_text is None:
                print("Response text is None. Aborting retries.")
                return None

            return response_text.strip()

        except Exception as e:
            # If the error is because response_text is None (and .strip() fails), do not retry.
            if "'NoneType' object has no attribute 'strip'" in str(e):
                print("Received None response text error. Not retrying further.")
                return None

            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Max retries reached. Returning None.")
                return None



def apply_wbf(boxes, original_resolution, iou_thr=0.5):
    if not boxes:
        return []

    # First scaling
    scaled_boxes = [[box[0] / original_resolution[1], box[1] / original_resolution[0],
                     box[2] / original_resolution[1], box[3] / original_resolution[0]]
                    for box in boxes]

    # Assign default scores and labels
    scores = [1.0] * len(scaled_boxes)
    labels = [1] * len(scaled_boxes)

    # Apply Weighted Box Fusion
    fused_boxes, _, _ = weighted_boxes_fusion(
        [scaled_boxes], [scores], [labels],
        iou_thr=iou_thr, skip_box_thr=0.0
    )

    # Rounding
    return [[round(coord, 3) for coord in box] for box in fused_boxes]



def custom_collate_fn(batch):
    # Ensure all items in the batch are tensors
    batch = [item for item in batch if item is not None]
    return batch



def get_img_transforms_mimic(img_size):
    transform = []
    transform.append(transforms.Resize(img_size))
    transform.append(transforms.CenterCrop(img_size))
    transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    transform.append(transforms.ToDtype(torch.float32, scale=True))
    transform.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    transform = transforms.Compose(transform)

    return transform




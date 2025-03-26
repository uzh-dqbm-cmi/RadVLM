import pydicom
from PIL import Image
import numpy as np
import os
import json  # For saving the dictionary as a JSON file
from pydicom.pixel_data_handlers import gdcm_handler, pylibjpeg_handler
import sys
import torch
import torchvision.transforms.v2 as transforms
import pandas as pd

def dcm2jpg_resolutions_vindrcxr(data_dir, split, image_size=512):
    """
    Convert DICOM images in a specified subdirectory of data_dir to JPEG format,
    saving the output in a similarly named subdirectory with "_jpg" appended.
    Also, save the resolutions of the images in a JSON file.

    Args:
    data_dir (str): The base directory of the dataset.
    split (str): The subdirectory within data_dir that contains the DICOM files.
    """

    input_dir = os.path.join(data_dir, split)
    output_dir = os.path.join(data_dir, split + "_jpg")
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to hold image IDs and resolutions
    resolution_dict = {}

    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.lower().endswith(".dicom"):
            dicom_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, os.path.splitext(filename)[0] + ".jpg"
            )

            # Check if JPEG file already exists, if so, skip
            if os.path.exists(output_path):
                continue

            try:
                # Read the DICOM file
                ds = pydicom.dcmread(dicom_path)

                # Store the resolution in the dictionary as a list [Rows, Columns]
                image_id = os.path.splitext(os.path.basename(dicom_path))[0]
                resolution_dict[image_id] = [ds.Rows, ds.Columns]

                # Get the pixel array from the DICOM dataset
                image = ds.pixel_array

                # Normalize and convert to 8-bit if necessary
                if image.dtype == np.uint16:
                    image = (
                        (image - np.min(image)) / (np.max(image) - np.min(image))
                    ) * 255.0
                    image = image.astype(np.uint8)

                # Convert to PIL image
                im = Image.fromarray(image)

                # Resize the image if necessary
                im_resized = im.resize(
                    (image_size, image_size), Image.Resampling.LANCZOS
                )
                # Save the resized image as JPEG
                im_resized.save(output_path)
            except Exception as e:
                print(f"Error processing {dicom_path}: {e}")

            # Print progress
            print(f"Processed {idx + 1}/{len(os.listdir(input_dir))}: {filename}")

    with open(os.path.join(data_dir, "image_resolutions_" + split + ".json"), "w") as f:
        json.dump(resolution_dict, f)

    print("Conversion and resolution extraction complete.")



DATA_DIR = os.environ.get('DATA_DIR')
if DATA_DIR is None:
    raise EnvironmentError("The environment variable 'DATA_DIR' is not set.")

vindr_cxr_data_dir = os.path.join(DATA_DIR, 'VinDr-CXR')

dcm2jpg_resolutions_vindrcxr(vindr_cxr_data_dir, 'train', image_size=512)
dcm2jpg_resolutions_vindrcxr(vindr_cxr_data_dir, 'test', image_size=512)

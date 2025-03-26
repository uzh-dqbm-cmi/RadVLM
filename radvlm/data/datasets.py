import os
import os.path
import sys

import tarfile
import xml
import numpy as np
import pandas as pd
from PIL import Image
import string
from skimage.io import imread
import re
import random
from torchvision.io import read_image
from sklearn.model_selection import train_test_split

import json
from torchxrayvision.datasets import (
    Dataset,
    normalize,
    USE_INCLUDED_FILE,
)
from PIL import Image
from radvlm.data.create_instructions import *
from radvlm.data.utils import *



class PadChest_grounding(Dataset):
    def __init__(
        self, 
        datasetpath, 
        split='train', 
        flag_img=True, 
        flag_instr=True,
        flag_txt=True,
    ):
        """
        datasetpath: Path to the folder that contains:
          - grounded_reports_20240819.json
          - master_table.csv
          - PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv (reports file)
          - images_grounding/ (subdirectory with images)
        split: "train", "validation", or "test" (or other if you have more)
        flag_img: If True, __getitem__ will load and return the image array.
        flag_instr: If True, additional instructions will be generated in __getitem__.
        flag_txt: If True, include the Spanish report text from the reports CSV.
        """
        super().__init__()
        self.datasetpath = datasetpath
        self.split = split
        self.flag_img = flag_img
        self.flag_instr = flag_instr
        self.flag_txt = flag_txt
        
        # 1) Read the master_table.csv and filter by split.
        master_table_path = os.path.join(self.datasetpath, 'master_table.csv')
        df_master = pd.read_csv(master_table_path)
        if split == 'valid':
            split = 'validation'
        df_split = df_master[df_master["split"] == split]

        # Map ImageID -> gender from the master table.
        imgid2gender = {}
        for _, row in df_split.iterrows():
            image_id = row["ImageID"]
            gender = row["PatientSex_DICOM"]
            imgid2gender[image_id] = gender
        
        # 2) Load reports CSV and filter by Projection ("AP" or "PA").
        reports_path = os.path.join(self.datasetpath, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
        df_reports = pd.read_csv(reports_path)
        df_reports = df_reports[df_reports["Projection"].isin(["AP", "PA"])]

        # Map ImageID -> Spanish report (column "Report").
        imgid2report = {}
        for _, row in df_reports.iterrows():
            image_id = row["ImageID"]
            imgid2report[image_id] = row["Report"]
        
        # 3) Read the JSON with grounded reports.
        grounding_reports_path = os.path.join(self.datasetpath, 'grounded_reports_20240819.json')
        with open(grounding_reports_path, 'r') as f:
            data = json.load(f)

        # 4) Flatten into self.samples, filtering out entries that are not in the chosen split,
        #    do not have a report with the correct projection, or have an empty "boxes" field.
        self.samples = []
        for entry in data:
            image_id = entry["ImageID"]
            # Skip if image_id is not in both the master table (split) and reports (filtered by Projection)
            if (image_id not in imgid2gender) or (image_id not in imgid2report):
                continue
            
            for finding in entry.get("findings", []):
                boxes = finding.get("boxes", [])
                # Skip datapoints with empty boxes.
                if not boxes:
                    continue

                self.samples.append({
                    "img_path": os.path.join(self.datasetpath, 'images_grounding', image_id),
                    "phrase": finding["sentence_en"],
                    "boxes": boxes,
                    "gender": imgid2gender[image_id],
                    "txt": imgid2report.get(image_id, "")
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a dictionary that includes:
          - "img_path": absolute path to the image.
          - "label": the English sentence (finding text).
          - "boxes": list of bounding box coordinates (each box as a list of 4 numbers).
          - "gender": the patient’s gender.
          - "txt": the Spanish report text (if flag_txt is True).
          - "img": image array (if flag_img is True).
          - "instr": additional instructions (if flag_instr is True).
        """
        sample_info = self.samples[idx]
        
        sample = {
            "img_path": sample_info["img_path"],
            "label": sample_info["phrase"],
            "boxes": sample_info["boxes"],
            "gender": sample_info["gender"],
        }

        if self.flag_txt:
            sample["txt"] = sample_info["txt"]

        if self.flag_img:
            img = imread(sample_info["img_path"])
            sample["img"] = normalize(img, maxval=255, reshape=True)
        
        if self.flag_instr:
            sample["instr"] = generate_instruction_phrase_location(
                sample["boxes"], sample["label"]
            )
        
        return sample




class PadChest_grounding_per_image(Dataset):
    def __init__(
        self, 
        datasetpath, 
        split='train', 
        flag_img=True, 
        flag_instr=True,
        flag_txt=True,
        conversation_dir=None,   # New argument for conversation files
    ):
        """
        datasetpath: Path to the folder that contains:
          - grounded_reports_20240819.json
          - master_table.csv
          - PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv (reports file)
          - images_grounding/ (subdirectory with images)
        split: "train", "validation", or "test" (or other if you have more)
        flag_img: If True, __getitem__ will load and return the image array.
        flag_instr: If True, additional instructions will be generated in __getitem__.
        flag_txt: If True, include the Spanish report text from the reports CSV.
        conversation_dir: Directory containing conversation JSON files.
                          Files should be named like the image files but with .json extension.
                          If provided, only samples with a corresponding conversation file will be used.
        """
        super().__init__()
        self.datasetpath = datasetpath
        self.split = split
        self.flag_img = flag_img
        self.flag_instr = flag_instr
        self.flag_txt = flag_txt
        self.conversation_dir = conversation_dir

        # 1) Read the master_table.csv and filter by split.
        master_table_path = os.path.join(self.datasetpath, 'master_table.csv')
        df_master = pd.read_csv(master_table_path)
        if split == 'valid':
            split = 'validation'
        df_split = df_master[df_master["split"] == split]

        # Map ImageID -> gender from the master table.
        imgid2gender = df_split.set_index("ImageID")["PatientSex_DICOM"].to_dict()
        
        # 2) Load reports CSV and filter by Projection ("AP" or "PA").
        reports_path = os.path.join(self.datasetpath, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
        df_reports = pd.read_csv(reports_path)
        df_reports = df_reports[df_reports["Projection"].isin(["AP", "PA"])]

        # Map ImageID -> Spanish report (column "Report") and Projection ("view")
        imgid2report = df_reports.set_index("ImageID")["Report"].to_dict()
        imgid2view = df_reports.set_index("ImageID")["Projection"].to_dict()

        # 3) Read the JSON with grounded reports.
        grounding_reports_path = os.path.join(self.datasetpath, 'grounded_reports_20240819.json')
        with open(grounding_reports_path, 'r') as f:
            data = json.load(f)

        # 4) Aggregate findings per ImageID.
        image_to_findings = {}
        for entry in data:
            image_id = entry["ImageID"]
            # Skip if image_id is not in both the master table (split) and reports (filtered by Projection)
            if (image_id not in imgid2gender) or (image_id not in imgid2report):
                continue
            
            findings = entry.get("findings", [])
            for finding in findings:
                boxes = finding.get("boxes", [])
                # Skip datapoints with empty boxes.
                if not boxes:
                    continue
                
                if image_id not in image_to_findings:
                    image_to_findings[image_id] = []
                
                for box in boxes:
                    # Append each phrase and box pair.
                    image_to_findings[image_id].append({
                        "observation": finding["sentence_en"],
                        "box": box
                    })
        
        # 5) Create self.samples with one entry per image.
        #    If conversation_dir is provided, we only keep samples that have a corresponding conversation file.
        self.samples = []
        for image_id, sentences_boxes in image_to_findings.items():
            # Construct the expected image file path. (Assuming images have an extension, e.g., .png)
            img_filename = image_id  # if image_id is the full filename like "example.png"
            img_path = os.path.join(self.datasetpath, 'images_grounding', img_filename)
            
            # If conversation_dir is provided, construct the corresponding conversation filename by replacing the extension.
            conversation = None
            if self.conversation_dir is not None:
                base, _ = os.path.splitext(img_filename)
                conversation_file = os.path.join(self.conversation_dir, base + '.json')
                if not os.path.exists(conversation_file):
                    # Skip this sample if the conversation file is not found.
                    continue
                else:
                    # Load conversation JSON now or later in __getitem__
                    # Here, we store the path and load during __getitem__
                    conversation = conversation_file

            sample = {
                "img_path": img_path,
                "sentencesBBox": sentences_boxes,
                "gender": imgid2gender[image_id],
                "view": imgid2view.get(image_id, None),
            }
            if self.flag_txt:
                sample["txt"] = imgid2report.get(image_id, "")
            if conversation is not None:
                sample["conversation_file"] = conversation  # store conversation file path

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a dictionary that includes:
          - "img_path": absolute path to the image.
          - "sentencesBBox": list of dictionaries with "observation" and "box".
          - "gender": the patient’s gender.
          - "view": the projection (AP or PA).
          - "txt": the Spanish report text (if flag_txt is True).
          - "img": image array (if flag_img is True).
          - "instr": additional instructions (if flag_instr is True).
          - "conversation": parsed JSON content from the corresponding conversation file (if conversation_dir is provided).
        """
        sample_info = self.samples[idx]
        
        sample = {
            "img_path": sample_info["img_path"],
            "sentencesBBox": sample_info["sentencesBBox"],
            "gender": sample_info["gender"],
            "view": sample_info["view"]
        }

        if self.flag_txt:
            sample["txt"] = sample_info["txt"]

        if self.flag_img:
            img = imread(sample_info["img_path"])
            sample["img"] = normalize(img, maxval=255, reshape=True)
        
        if self.flag_instr:
            # Insert your instruction generation logic if desired.
            instructions = []
            for item in sample["sentencesBBox"]:
                # Example placeholder: generate instruction based on box and observation.
                # instr = generate_instruction_phrase_location(item["box"], item["observation"])
                # instructions.append(instr)
                pass
            sample["instr"] = instructions

        # If a conversation file is specified, load the conversation.
        if self.conversation_dir is not None:
            conv_file = sample_info.get("conversation_file", None)
            if conv_file is not None and os.path.exists(conv_file):
                with open(conv_file, 'r') as cf:
                    sample["conversation"] = json.load(cf)
            else:
                sample["conversation"] = None

        return sample




class CheXpert_Dataset_MM(Dataset):
    """For CheXpert dataset"""

    def __init__(
        self,
        datasetpath,
        split="train",
        only_frontal=True,
        flag_img=True,
        flag_instr=True,
        flag_lab=True,
        unique_patients=False,
        seed=0,
    ):  
        self.datasetpath = datasetpath 
        train_csv_path = os.path.join(datasetpath, "train.csv")
        valid_csv_path = os.path.join(datasetpath, "valid.csv")
        test_csv_path = os.path.join(datasetpath, "test.csv")

        if split == "train":
            self.csv = pd.read_csv(train_csv_path)
        elif split == "valid":
            self.csv = pd.read_csv(valid_csv_path)
        elif split == "test":
            self.csv = pd.read_csv(test_csv_path)
        else:
            raise ValueError(f"The value of split '{split}' is incorrect. Expected 'train', 'valid', or 'test'.")
        
        # Filter for frontal views if only_frontal is True
        if only_frontal:
            self.csv = self.csv[self.csv["Path"].str.contains(r'frontal\.jpg$')]

        self.flag_img = flag_img
        self.flag_instr = flag_instr
        self.flag_lab = flag_lab

        self.pathologies = sorted([
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", 
            "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
            "Pneumothorax", "Pleural Effusion", "Pleural Other", 
            "Fracture", "Support Devices"
        ])


        if split in ['valid', 'train']:
            self.csv['age_years'] = self.csv['Age'] * 1.0
            self.csv.loc[self.csv['Age'] == 0, 'Age'] = None

            # Encode sex
            self.csv['sex_male'] = self.csv['Sex'] == 'Male'
            self.csv['sex_female'] = self.csv['Sex'] == 'Female'

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(r'(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Create labels array
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            labels.append(mask.values)
        self.labels = np.asarray(labels).T.astype(np.float32)
        self.labels[self.labels == -1] = 1

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        sample = {"idx": idx, "labels": self.labels[idx]}
        imgid = self.csv["Path"].iloc[idx]
        imgid = os.path.join(self.datasetpath, imgid)
        img_path = imgid.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "") 

        if self.flag_img:
            img = imread(img_path)
            sample["img"] = normalize(img, maxval=255, reshape=True)

        sample["img_path"] = img_path

        if self.flag_lab:
            labels = [self.pathologies[i] for i, v in enumerate(sample["labels"]) if v == 1]
            sample["labels"] = labels
        if self.flag_instr:
            sample["instr"] = generate_instruction_abnormalities(sample["labels"])
        
        return sample
        




class VinDr_CXR_Dataset(Dataset):
    """For VinDr_CXR and VinDr-PCXR.
    Download https://physionet.org/content/vindr-cxr/ and https://physionet.org/content/vindr-pcxr/
    """

    def __init__(self, datasetpath, split="train", flag_img=True, flag_instr=True, seed=0):
        super(Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasetpath = datasetpath
        self.flag_img = flag_img
        self.flag_instr = flag_instr

        annotations_dir = "annotations" if os.path.isdir(os.path.join(self.datasetpath, "annotations")) else ""

        if split in ["train", "test"]:
            original_split = "train" if split == "train" else "test"

            self.imgpath = os.path.join(self.datasetpath, f"{original_split}_jpg")

            resolutions_path = os.path.join(
                self.datasetpath, annotations_dir, f"image_resolutions_{original_split}.json"
            )
            if not os.path.exists(resolutions_path):
                raise ValueError("The image resolutions file cannot be found.")
            with open(resolutions_path, "r") as file:
                self.resolutions = json.load(file)
            
            annotations_path = os.path.join(
                self.datasetpath, annotations_dir, f"annotations_{original_split}.csv"
            )
            self.annotations = pd.read_csv(annotations_path)
            self.annotations.rename(columns={"rad_ID": "rad_id"}, inplace=True)

            # Group annotations by image_id and class_name
            grouped_annotations = self.annotations.groupby(["image_id", "class_name"])

            # Apply WBF for each group
            fused_annotations = []
            for (image_id, class_name), group in grouped_annotations:
                if class_name != "No finding":
                    boxes = group[['x_min', 'y_min', 'x_max', 'y_max']].dropna().values.tolist()
                    original_resolution = self.resolutions.get(image_id, [1024, 1024])  # Default resolution if not found
                    if len(boxes) > 0:
                        fused_boxes = apply_wbf(boxes, original_resolution, iou_thr=0.1)
                        for box in fused_boxes:
                            fused_annotations.append([image_id, class_name, *box])
                else:
                    fused_annotations.append([image_id, class_name, None, None, None, None])

            self.annotations = pd.DataFrame(
                fused_annotations, 
                columns=['image_id', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max']
            )
        else:
            raise ValueError(f"The value of split '{split}' is incorrect. Expected 'train' or 'test'.")

        self.image_to_annotations = self.annotations.set_index("image_id")
        self.image_files = self.annotations['image_id'].unique().tolist()
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        sample = {}
        image_id = str(self.image_files[idx])
        img_filename = image_id + ".jpg"
        imgpath = os.path.join(self.imgpath, img_filename)
        sample["img_path"] = imgpath

        if self.flag_img:
            img = imread(imgpath)
            sample["img"] = normalize(img, maxval=255, reshape=True)

        if image_id in self.image_to_annotations.index:
            image_annotations = self.image_to_annotations.loc[image_id]
            if not isinstance(image_annotations, pd.DataFrame):
                image_annotations = image_annotations.to_frame().T  # Handle single row

            bounding_boxes = []
            class_labels = []
            for _, row in image_annotations.iterrows():
                if pd.notna(row["x_min"]):
                    x_min, y_min, x_max, y_max = row[["x_min", "y_min", "x_max", "y_max"]]
                    bounding_boxes.append([x_min, y_min, x_max, y_max])
                    class_labels.append(row["class_name"])

            if not bounding_boxes:  # No abnormalities detected
                bounding_boxes = []
                class_labels = ["No finding"]
        else:
            bounding_boxes = []
            class_labels = ["No finding"]

        sample["boxes"] = bounding_boxes
        sample["labels"] = class_labels

        if self.flag_instr:
            sample["instr"] = generate_instruction_abnormalities_grouped(bounding_boxes, sample["labels"])

        return sample



class VinDr_CXR_Single_Label_Dataset(Dataset):
    """
    A dataset that applies WBF per label and creates separate data points for each label found
    in an image.
    """

    def __init__(self, datasetpath, split="train", flag_img=True, flag_instr=True, seed=0):
        super(Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasetpath = datasetpath
        self.flag_img = flag_img
        self.flag_instr = flag_instr

        annotations_dir = "annotations" if os.path.isdir(os.path.join(self.datasetpath, "annotations")) else ""

        if split in ["train", "test"]:
            original_split = "train" if split == "train" else "test"

            self.imgpath = os.path.join(self.datasetpath, f"{original_split}_jpg")

            resolutions_path = os.path.join(
                self.datasetpath, annotations_dir, f"image_resolutions_{original_split}.json"
            )
            if not os.path.exists(resolutions_path):
                raise ValueError("The image resolutions file cannot be found.")
            with open(resolutions_path, "r") as file:
                self.resolutions = json.load(file)
            
            annotations_path = os.path.join(
                self.datasetpath, annotations_dir, f"annotations_{original_split}.csv"
            )
            self.annotations = pd.read_csv(annotations_path)
            self.annotations.rename(columns={"rad_ID": "rad_id"}, inplace=True)

            # Group annotations by image_id and class_name (label)
            grouped_annotations = self.annotations.groupby(["image_id", "class_name"])

            self.single_label_metadata = []

            # Apply WBF for each group (image_id + class_name)
            for (image_id, class_name), group in grouped_annotations:
                if class_name != "No finding":  # Ignore "No finding" cases
                    boxes = group[['x_min', 'y_min', 'x_max', 'y_max']].dropna().values.tolist()
                    original_resolution = self.resolutions.get(image_id, [1024, 1024])  # Default resolution if not found
                    if boxes:
                        fused_boxes = apply_wbf(boxes, original_resolution, iou_thr=0.1)
                        # Append separate entries for each label and fused bounding boxes
                        self.single_label_metadata.append((image_id, class_name, fused_boxes))
        else:
            raise ValueError(f"The value of split '{split}' is incorrect. Expected 'train' or 'test'.")

        # Create a list of image_ids for quick access
        self.image_files = [entry[0] for entry in self.single_label_metadata]
        
    def __len__(self):
        return len(self.single_label_metadata)

    def __getitem__(self, idx):
        image_id, label, fused_boxes = self.single_label_metadata[idx]
        img_filename = image_id + ".jpg"
        imgpath = os.path.join(self.imgpath, img_filename)

        sample = {
            "idx": idx,
            "img_path": imgpath,
            "label": label,  # The label for this datapoint
            "boxes": fused_boxes,  # Bounding boxes associated with this label
        }

        # Add image data if required
        if self.flag_img:
            img = imread(imgpath)
            sample["img"] = normalize(img, maxval=255, reshape=True)

        # Generate instructions if required
        if self.flag_instr:
            sample["instr"] = generate_instruction_location(fused_boxes, label)

        return sample
    



class MIMIC_Dataset_MM(Dataset):
    """MIMIC-CXR Dataset with img and text reports
    takes the split csv file to get the original train/val/test splits from MIMIC-CXR
    applies a text transformation with the function text_extraction_transform to extract out the findings (todo: set it as an option)

    """

    def __init__(
        self,
        datasetpath,
        split="train",
        flag_img=True,
        flag_txt=True,
        flag_instr=True,
        flag_lab=True,
        seed=0,
        img_size=224,
        only_frontal=False,  # Added parameter
        filtered_reports_dir=None,
        sentencesBBoxpath=None,
        conversation_dir=None,
        genderpath = None,
        classif=False
    ):

        super(Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.imgpath = os.path.join(datasetpath, 'files')
        self.img_size = img_size
        # self.img_transforms = get_img_transforms_mimic(self.img_size)

        self.flag_img = flag_img
        self.flag_txt = flag_txt
        self.flag_instr = flag_instr
        self.flag_lab = flag_lab
        self.only_frontal = only_frontal  # Store the parameter
        self.classif = classif

        self.splitcsvpath = os.path.join(datasetpath, 'mimic-cxr-2.0.0-split.csv')
        self.splitcsv = pd.read_csv(self.splitcsvpath)
        self.csvpath =  os.path.join(datasetpath, 'mimic-cxr-2.0.0-chexpert.csv')
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = os.path.join(datasetpath, 'mimic-cxr-2.0.0-metadata.csv')
        self.metacsv = pd.read_csv(self.metacsvpath)

        self.pathologies = sorted([
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", 
            "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
            "Pneumothorax", "Pleural Effusion", "Pleural Other", 
            "Fracture", "Support Devices"
        ])

        self.reportspath = os.path.join(datasetpath, 'reports.csv')
        self.reports = pd.read_csv(self.reportspath)

        # Remove the 's' prefix and convert 'study' to integer in self.reports
        self.reports['study'] = self.reports['study'].str.lstrip('s').astype(int)

        # Set index for metacsv and splitcsv, then join them
        self.metacsv.set_index(["dicom_id", "subject_id", "study_id"], inplace=True)
        self.splitcsv.set_index(["dicom_id", "subject_id", "study_id"], inplace=True)
        self.splitcsv = self.splitcsv[self.splitcsv.index.isin(self.metacsv.index)]
        self.metacsv = self.metacsv.join(self.splitcsv).reset_index()

        # Set index for self.csv and metacsv, then join them
        self.csv.set_index(["subject_id", "study_id"], inplace=True)
        self.metacsv.set_index(["subject_id", "study_id"], inplace=True)
        self.csv = self.csv.join(self.metacsv).reset_index()

        # Now process self.reports and join it with self.csv
        # Convert study in reports to integer and set it as index for joining
        self.reports.set_index("study", inplace=True)

        # Join the reports with self.csv based on study_id, keeping 'study_id' as a column
        self.csv = self.csv.join(self.reports, how='inner', on="study_id")  # No reset_index needed

        if split == "train":
            self.csv = self.csv[self.csv["split"] == "train"].drop(columns="split")
        elif split == "valid":
            self.csv = self.csv[self.csv["split"] == "validate"].drop(columns="split")
        elif split == "test":
            self.csv = self.csv[self.csv["split"] == "test"].drop(columns="split")
        else:
            # If the split is not one of the expected values, raise a ValueError
            raise ValueError(
                f"The value of split '{split}' is incorrect. Expected 'train', 'valid', or 'test'."
            )

        # Filter to include only frontal views if only_frontal is True
        if self.only_frontal:
            self.csv = self.csv[self.csv['ViewPosition'].isin(['PA', 'AP'])]

        
        self.filtered_reports_dir = filtered_reports_dir

        if self.filtered_reports_dir is not None:
            # Get all existing study_id files in the directory as a set
            existing_files = set(f.split(".txt")[0] for f in os.listdir(self.filtered_reports_dir) if f.endswith(".txt"))
            
            # Filter the CSV only for those rows where the study_id exists in the set
            self.csv = self.csv[self.csv['study_id'].astype(str).isin(existing_files)]


        # Get our classes.
        if self.pathologies is not None:
            healthy = self.csv["No Finding"] == 1
            self.csv = self.csv.fillna(0)
            self.csv.replace(-1, 1, inplace=True)
            labels = []
            for pathology in self.pathologies:
                if pathology in self.csv.columns:
                    # self.csv.loc[healthy, pathology] = 0
                    # mask = self.csv[pathology]
                    labels.append(self.csv[pathology].values)
            self.labels = np.asarray(labels).T
            self.labels = self.labels.astype(np.float32)

        
        self.sentencesBBoxpath = sentencesBBoxpath

        self.conversation_dir = conversation_dir

        if self.conversation_dir is not None:
            self.csv = self.csv[self.csv['dicom_id'].isin([f.replace('.json', '') for f in os.listdir(self.conversation_dir)])]
        

        if self.sentencesBBoxpath is not None:
            filenames = [f.replace('.json', '') for f in os.listdir(self.sentencesBBoxpath)]
            self.csv = self.csv[self.csv['dicom_id'].isin(filenames)]

        self.gender_json_path = genderpath
        if self.gender_json_path is not None:
            with open(self.gender_json_path, 'r') as file:
                self.genders_dict = json.load(file)
        else:
            self.genders_dict = None


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        sample["study_id"] = studyid
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(
            self.imgpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id + ".jpg",
        )
        sample["img_path"] = img_path

        if self.flag_img:
            img = imread(img_path)
            img = normalize(img, maxval=255, reshape=True)
            # img = read_image(img_path)
            # sample["img"] = self.img_transforms(img)
            sample["img"] = img

        sample["txt"] = np.nan  # Allows for single modality
        sample["instr"] = np.nan

        sample["view"] = self.csv.iloc[idx]["ViewPosition"]

        sample["gender"] = None  # default to None
        if self.genders_dict is not None:
            sample["gender"] = self.genders_dict.get(dicom_id, None)  # safe lookup

        if self.flag_txt or self.flag_instr:
            if self.filtered_reports_dir is None:  # Get the reports directly from self.csv
                # Directly use idx to access the row as a Series
                study_row = self.csv.iloc[idx]  # Fetch the row using idx directly

                text_parts = []
                # Check each report-related column and add it to the list if it's not NaN
                if pd.notna(study_row['findings']) and study_row['findings'] != 0:
                    text_parts.append(str(study_row['findings']))
                elif pd.notna(study_row['impression']) and study_row['impression'] != 0:
                    text_parts.append(str(study_row['impression']))
                elif pd.notna(study_row['last_paragraph']) and study_row['last_paragraph'] != 0:
                    text_parts.append(str(study_row['last_paragraph']))
                # Join the non-NaN parts into a single text string
                txt = "".join(text_parts).replace("\n", "")
            else:
                # Get the reports from the generated directory if available
                txt_path = os.path.join(self.filtered_reports_dir, str(studyid) + ".txt")
                with open(txt_path, "r") as file:
                    txt = file.read()

            if self.flag_txt:
                sample["txt"] = txt
            if self.flag_instr:
                sample["instr"] = generate_instruction_report_generation(txt)  

        if self.flag_lab:
            sample["labels"] = [self.pathologies[i] for i, v in enumerate(self.labels[idx]) if (v == 1 or v == -1)]
            if self.flag_instr and self.classif:
                sample["instr"] = generate_instruction_abnormalities(sample["labels"]) 
        
        sample["sentencesBBox"] = None 
        if self.sentencesBBoxpath is not None:
            sentencesBBox_path = os.path.join(self.sentencesBBoxpath, dicom_id + ".json")
            if os.path.exists(sentencesBBox_path):
                with open(sentencesBBox_path, 'r') as file:
                    sample["sentencesBBox"] = json.load(file)

        
        if self.conversation_dir is not None:
            conversation_path = os.path.join(self.conversation_dir, dicom_id + ".json")
            if os.path.exists(conversation_path):
                with open(conversation_path, 'r') as file:
                    sample["conversation"] = json.load(file)
            else:
                print(conversation_path)

        return sample



class Chest_ImaGenome_Dataset(MIMIC_Dataset_MM):
    """Extension of MIMIC-CXR with additional attributes, like bounding boxes for anatomical regions
    Available for download here: https://www.physionet.org/content/chest-imagenome/1.0.0/
    We add other attributes
    :sentencesBBoxpath: the directory containing the generated sentences-BBox associations from llama
    :conversation_dir: the directory containing generated conversations from llama. When one, length of dataset 
    determined by number of files contained in this directory
    """

    def __init__(
        self,
        datasetpath_chestima,
        pick_one_region=True,
        split = 'train',
        sentencesBBoxpath=None,
        conversation_dir=None,
        *args, 
        **kwargs
    ):
        """
        :datasetpath: path to Chest ImaGenome directory (assuming content from physionet)
        """
        super().__init__(*args, **kwargs)

        self.pick_one_region = pick_one_region

        self.splits_path = os.path.join(datasetpath_chestima, "silver_dataset/splits")
        self.scene_graph_path = os.path.join(datasetpath_chestima, "silver_dataset/scene_graph")

        self.pathologies = sorted([
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", 
            "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
            "Pneumothorax", "Pleural Effusion", "Pleural Other", 
            "Fracture", "Support Devices"
        ])



        if split == "train":
            self.splitcsv_chestima = pd.read_csv(os.path.join(self.splits_path, "train.csv"))
        elif split == "valid":
            self.splitcsv_chestima = pd.read_csv(os.path.join(self.splits_path, "valid.csv"))
        elif split == "test":
            self.splitcsv_chestima = pd.read_csv(os.path.join(self.splits_path, "test.csv"))
        else:
            # If the split is not one of the expected values, raise a ValueError
            raise ValueError(
                f"The value of split '{split}' is incorrect. Expected 'train', 'valid', or 'test'."
            )

        # Check for missing ids in scene_graph directory (actually observed from original dataset)
        missing_ids_path = os.path.join(datasetpath_chestima, "silver_dataset/missing_ids.json")
        if not os.path.exists(missing_ids_path):
            missing_ids = []
            for filename in os.listdir(self.splits_path):
                if filename.endswith(".csv"):
                    csv_path = os.path.join(self.splits_path, filename)
                    data = pd.read_csv(csv_path)
                    for dicom_id in data["dicom_id"]:
                        scene_graph_file = os.path.join(
                            self.scene_graph_path, f"{dicom_id}_SceneGraph.json"
                        )
                        if not os.path.isfile(scene_graph_file):
                            missing_ids.append(dicom_id)
            json_data = json.dumps(missing_ids)
            with open(missing_ids_path, "w") as outfile:
                outfile.write(json_data)
                print("Missing ids dictionary saved")
        else:
            # Load missing IDs from JSON file
            with open(missing_ids_path, "r") as f:
                missing_ids = json.load(f)

        # Filter self.splitcsv to exclude the IDs in missing_ids
        self.splitcsv_chestima = self.splitcsv_chestima[~self.splitcsv_chestima["dicom_id"].isin(missing_ids)]

        self.splitcsv_chestima.set_index(["dicom_id", "subject_id", "study_id", "ViewPosition"], inplace=True)
        self.csv.set_index(["dicom_id", "subject_id", "study_id", "ViewPosition"], inplace=True)

        self.csv = self.csv[self.csv.index.isin(self.splitcsv_chestima.index)]
        self.csv.reset_index(inplace=True)

        self.sentencesBBoxpath = sentencesBBoxpath

        self.conversation_dir = conversation_dir

        if self.conversation_dir is not None:
            self.csv = self.csv[self.csv['dicom_id'].isin([f.replace('.json', '') for f in os.listdir(self.conversation_dir)])]
        

        if self.sentencesBBoxpath is not None:
            filenames = [f.replace('.json', '') for f in os.listdir(self.sentencesBBoxpath)]
            self.csv = self.csv[~self.csv['dicom_id'].isin(filenames)]

                # Get our classes.
        if self.pathologies is not None:
            healthy = self.csv["No Finding"] == 1
            self.csv = self.csv.fillna(0)
            self.csv.replace(-1, 1, inplace=True)
            labels = []
            for pathology in self.pathologies:
                if pathology in self.csv.columns:
                    labels.append(self.csv[pathology].values)
            self.labels = np.asarray(labels).T
            self.labels = self.labels.astype(np.float32)


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(
            self.imgpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id + ".jpg",
        )
        sample["img_path"] = img_path

        if self.flag_img:
            img = imread(img_path)
            img = normalize(img, maxval=255, reshape=True)
            sample["img"] = img

        sample["txt"] = np.nan  # Allows for single modality
        sample["instr"] = np.nan

        if self.filtered_reports_dir is None:  # Get the reports directly from self.csv
            # Directly use idx to access the row as a Series
            study_row = self.csv.iloc[idx]  # Fetch the row using idx directly

            text_parts = []
            # Check each report-related column and add it to the list if it's not NaN
            if pd.notna(study_row['findings']) and study_row['findings'] != 0:
                text_parts.append(str(study_row['findings']))
            elif pd.notna(study_row['impression']) and study_row['impression'] != 0:
                text_parts.append(str(study_row['impression']))
            elif pd.notna(study_row['last_paragraph']) and study_row['last_paragraph'] != 0:
                text_parts.append(str(study_row['last_paragraph']))
            # Join the non-NaN parts into a single text string
            txt = "".join(text_parts).replace("\n", "")
        else:
            # Get the reports from the generated directory if available
            txt_path = os.path.join(self.filtered_reports_dir, str(studyid) + ".txt")
            with open(txt_path, "r") as file:
                txt = file.read()

        if self.flag_txt:
            sample["txt"] = txt


        scene_graph_path = os.path.join(
            self.scene_graph_path, dicom_id + "_SceneGraph.json"
        )
        with open(scene_graph_path, "r") as file:  # Open the file before loading
            scene_graph = json.load(file)

        objects_data = scene_graph["objects"]
        width_img = self.csv.iloc[idx]["Columns"]
        height_img = self.csv.iloc[idx]["Rows"]

        if self.pick_one_region:
            random_object = random.choice(objects_data)
            bounding_box = [
                float(random_object["original_x1"]) / width_img,
                float(random_object["original_y1"]) / height_img,
                float(random_object["original_x2"]) / width_img,
                float(random_object["original_y2"]) / height_img,
            ]
            sample["boxes"] = [bounding_box]
            sample["label"] = random_object["bbox_name"]
        else:
            bounding_boxes = []
            region_names = []
            for obj in objects_data:
                bounding_boxes.append([
                    float(obj["original_x1"]) / width_img,
                    float(obj["original_y1"]) / height_img,
                    float(obj["original_x2"]) / width_img,
                    float(obj["original_y2"]) / height_img,
                ])
                region_names.append(obj["bbox_name"])
            sample["boxes"] = bounding_boxes
            sample["labels"] = region_names


        view = scene_graph["viewpoint"]
        sample["view"] = view

        gender = scene_graph["gender"]
        sample["gender"] = gender

        if self.flag_lab:
            labels = self.labels[idx]
            sample["labels"] = [self.pathologies[i] for i, label in enumerate(labels) if label == 1]

        
        if self.flag_instr:
            sample["instr"] = generate_instruction_location(
                sample["boxes"], sample["label"]
            )

        return sample



class MS_CXR(MIMIC_Dataset_MM):
    """
    A subclass of Chest_ImaGenome_Dataset that groups all bounding boxes
    for the same 'observation' into a single datapoint.
    
    That is, if 'cardiac silhouette is enlarged' appears multiple times 
    in the JSON with different bounding boxes, they all go into one sample.
    """

    def __init__(
        self,
        split='train',
        sentencesBBoxpath=None,
        *args,
        **kwargs
    ):
        # 1) Parent init: loads self.csv, sets up paths, flags, etc.
        super().__init__(
            split=split,
            *args,
            **kwargs
        )

        # 2) Build a flattened list of data points: 
        #    one per (image, single-phrase) but collecting multiple boxes if repeated.
        self.flattened_data = []
        self.sentencesBBoxpath = sentencesBBoxpath

        if self.sentencesBBoxpath is not None:
            filenames = [f.replace('.json', '') for f in os.listdir(self.sentencesBBoxpath)]
            self.csv = self.csv[self.csv['dicom_id'].isin(filenames)]

        if self.sentencesBBoxpath is not None:
            for i in range(len(self.csv)):
                dicom_id = str(self.csv.iloc[i]["dicom_id"])
                subject_id = str(self.csv.iloc[i]["subject_id"])
                study_id   = str(self.csv.iloc[i]["study_id"])

                # Construct the path to the image
                img_path = os.path.join(
                    self.imgpath,
                    "p" + subject_id[:2],
                    "p" + subject_id,
                    "s" + study_id,
                    dicom_id + ".jpg",
                )

                # Locate and load the sentencesBBox file for this dicom_id
                sbbox_file = os.path.join(self.sentencesBBoxpath, dicom_id + ".json")
                if os.path.exists(sbbox_file):
                    with open(sbbox_file, 'r') as f:
                        sbbox_data = json.load(f)

                    # Group bounding boxes by identical "observation"
                    boxes_by_obs = defaultdict(list)
                    for entry in sbbox_data:
                        observation = entry["observation"]
                        box = entry["box"]  # [x1, y1, x2, y2] (normalized)
                        boxes_by_obs[observation].append(box)

                    # For each unique observation, build a single record with all boxes
                    for obs, box_list in boxes_by_obs.items():
                        self.flattened_data.append({
                            "dicom_id": dicom_id,
                            "img_path": img_path,
                            "observation": obs,
                            "boxes": box_list  # collect all bounding boxes for that phrase
                        })
                # else, no JSON => skip or do nothing
        else:
            # If no sentencesBBoxpath, we won't have any data
            self.flattened_data = []

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        """
        Returns a sample that contains:
          - 'img_path'
          - 'img' (if flag_img)
          - 'sentencesBBox' (list of {"observation": ..., "box": ...} for all boxes)
          - 'boxes' (list of bounding boxes)
          - 'label' (the phrase text)
          - 'instr' (an instruction derived from the boxes and phrase, if flag_instr)
        """
        record = self.flattened_data[idx]

        sample = {}
        sample["img_path"] = record["img_path"]

        # 1) Load the image if flag_img is True
        if self.flag_img:
            img = imread(sample["img_path"])
            img = normalize(img, maxval=255, reshape=True)
            sample["img"] = img
        else:
            sample["img"] = None

        # 2) Rebuild a sentencesBBox-like list from all the boxes for that phrase
        #    so you can keep consistent naming with your original code
        sbbox_list = []
        for b in record["boxes"]:
            sbbox_list.append({
                "observation": record["observation"],
                "box": b
            })
        sample["sentencesBBox"] = sbbox_list

        # 3) 'boxes' => all bounding boxes for this phrase
        sample["boxes"] = record["boxes"]

        # 4) 'label' => the (unique) observation/phrase text
        sample["label"] = record["observation"]

        # 5) 'instr' => generate from all bounding boxes + phrase
        if self.flag_instr:
            sample["instr"] = generate_instruction_phrase_location(
                record["boxes"], record["observation"]
            )
        else:
            sample["instr"] = None

        return sample






class CheXpertPlus_Dataset(CheXpert_Dataset_MM):
    """
    223,228 unique pairs of radiology reports (and labels) and chest X-rays from 187,711 studies and 64,725 patients.
    Available for download: https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1
    """

    def __init__(
        self,
        datasetpath,
        split="train",
        flag_img=True,
        flag_txt=True,
        flag_instr=True,
        flag_lab=True,
        only_frontal=True,
        filtered_reports_dir=None,
        seed=0,
    ):
        super(CheXpert_Dataset_MM).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.datasetpath = datasetpath
        self.flag_img = flag_img
        self.flag_txt = flag_txt
        self.flag_instr = flag_instr
        self.flag_lab = flag_lab

        # Read the specified columns from the CSV file
        reports_csv = pd.read_csv(
            os.path.join(datasetpath, "df_chexpert_plus_240401.csv"),
        )

        # Filter the rows based on the split
        if split == "train":
            self.reports_csv = reports_csv[reports_csv["split"] == "train"]
        elif split == "valid":
            raise ValueError("Validation set is not available when using 100 percent for training.")
        elif split == "test":
            self.reports_csv = reports_csv[reports_csv["split"] == "valid"]
        else:
            # If the split is not one of the expected values, raise a ValueError
            raise ValueError(
                f"The value of split '{split}' is incorrect. Expected 'train', 'valid', or 'test'."
            )
        
        if only_frontal==True:
            self.reports_csv =  self.reports_csv[self.reports_csv['frontal_lateral'] == 'Frontal']
            

        # Open the JSON file for reading and create the labels dictionary
        labels_file = os.path.join(datasetpath, "chexbert_labels/report_fixed.json")

        self.filtered_reports_dir = filtered_reports_dir

        if filtered_reports_dir is not None:
            available_txt_files = [
                filename for filename in os.listdir(filtered_reports_dir)
            ]
            self.reports_csv = self.reports_csv[
                self.reports_csv['path_to_image'].apply(
                    lambda x: '_'.join(x.split('/')[:3]) + '.txt'
                ).isin(available_txt_files)
            ]
        self.labels = {}
        with open(labels_file, "r") as file:
            for line in file:
                record = json.loads(line)
                img_path = os.path.join(record.pop("path_to_image"))
                self.labels[img_path] = record

    def __len__(self):
        return len(self.reports_csv)

    def __getitem__(self, idx):
        sample = {}

        img_path = os.path.join(
            self.datasetpath,
            'PNG',
            str(self.reports_csv.iloc[idx]["path_to_image"]).replace(".jpg", ".png")
        )
        sample["img_path"] = img_path

        sample["img"] = np.nan

        if self.flag_img:
            img = imread(img_path)
            sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.flag_txt:
            if self.filtered_reports_dir is None:
                columns = [
                    "section_findings",
                    "section_impression",
                    "section_end_of_impression",
                ]
                report_parts = [
                    str(self.reports_csv.iloc[idx][col])
                    for col in columns
                    if pd.notna(self.reports_csv.iloc[idx][col])
                ]
                report = "".join(report_parts)
                sample["report"] = report

                sample["txt"] = report.capitalize().replace("\n", "")
            else:
                txt_path = "_".join(self.reports_csv.iloc[idx]["path_to_image"].split('/')[:3]) + ".txt"
                txt_path = os.path.join(self.filtered_reports_dir, txt_path)

                with open(txt_path, "r") as file:
                    sample["txt"] = file.read()


        if self.flag_lab:
            labels = [
                key
                for key, value in self.labels.get(self.reports_csv.iloc[idx]["path_to_image"], {}).items()
                if value == 1.0
            ]
            sample["labels"] = labels
        
        if self.flag_instr:
            sample["instr"] = generate_instruction_report_generation(
                sample["txt"]
            )  

        return sample

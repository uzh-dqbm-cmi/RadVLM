import torch
import numpy as np
from radvlm.data.datasets import MIMIC_Dataset_MM, CheXpert_Dataset_MM, Chest_ImaGenome_Dataset, MS_CXR, CheXpertPlus_Dataset, PadChest_grounding, PadChest_grounding_per_image, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset
from radvlm.data.utils import *
import json
import os
from torch.utils.data import ConcatDataset
from radvlm.data.create_instructions import generate_llava_dataset_from_instruction_dataset
from radvlm import DATA_DIR

# MIMIC-CXR 
print("MIMIC-CXR reports")
datasetpath_mimic = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports')
mimic_dataset_filtered = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
                                          split="train", 
                                          flag_img=False, flag_lab=False, 
                                          only_frontal=True, 
                                          filtered_reports_dir=filtered_reports_dir, 
                                          seed=0
                                         )
print("Num samples = " + str(len(mimic_dataset_filtered)))
print("")


print("MIMIC-CXR classif")
# MIMIC-CXR 
datasetpath_mimic = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
mimic_dataset_labels = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
                                          split="train", 
                                          flag_img=False, flag_lab=True, 
                                          only_frontal=True, 
                                          filtered_reports_dir=None, 
                                          classif=True,
                                          seed=0
                                         )
print("Num samples = " + str(len(mimic_dataset_labels)))
print("")

# CheXpert 
print("CheXpert classif")
dataset_path = os.path.join(DATA_DIR, "CheXpert")   
chexpert_dataset = CheXpert_Dataset_MM(datasetpath=dataset_path,split="train", flag_img=False)
print("Num samples = " + str(len(chexpert_dataset)))
print("")


#CheXpert-Plus reports
print("CheXpert reports")
datasetpath = os.path.join(DATA_DIR, 'CheXpertPlus')
filtered_reports_dir = os.path.join(datasetpath, 'filtered_reports')
chexpertplus_dataset = CheXpertPlus_Dataset(datasetpath=datasetpath, split='train', flag_img=False, filtered_reports_dir=filtered_reports_dir)
print("Num samples = " + str(len(chexpertplus_dataset)))
print("")


#CHEST_IMA
print("Chest-ima")
datasetpath_chestima = os.path.join(DATA_DIR, 'CHEST_IMA')
split = "train"
chestima_dataset = Chest_ImaGenome_Dataset(
    datasetpath=datasetpath_mimic,
    datasetpath_chestima=datasetpath_chestima,
    split=split, 
    flag_img=False, 
    flag_instr=True, 
    flag_txt=False, 
    flag_lab=False,
    pick_one_region=True,
    )
print("Num samples = " + str(len(chestima_dataset)))
print("")


# VinDr-CXR

print("VinDr-CXR")
dataset_path = os.path.join(DATA_DIR, "VinDr-CXR") 
vin_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split="train", flag_img = False)
print("Num samples = " + str(len(vin_dataset)))
print("")

vin_dataset_mono = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split="train", flag_img = False)
print("Num samples = " + str(len(vin_dataset_mono)))
print("")


# Phrase grounding 
print("Phrase grounding MS-CXR")

sentencesBBoxpath = os.path.join(DATA_DIR, 'MS-CXR','sentences_and_BBox_mscxr')


dataset_train = MS_CXR(
    datasetpath = datasetpath_mimic,
    split="train", flag_img=False, 
    flag_lab=True, only_frontal=True, 
    flag_instr=True, 
    sentencesBBoxpath=sentencesBBoxpath,
    seed=0)

dataset_valid = MS_CXR(
    datasetpath = datasetpath_mimic,
    split="valid", flag_img=False, 
    flag_lab=True, only_frontal=True, 
    flag_instr=True, 
    sentencesBBoxpath=sentencesBBoxpath,
    seed=0)


prhase_grounding_mscxr_dataset = ConcatDataset([dataset_train, dataset_valid])

print("Num samples = " + str(len(prhase_grounding_mscxr_dataset)))
print("")


print("Phrase grounding PadChest")

datasetpath = os.path.join(DATA_DIR, 'PadChest')
split = 'train' 
dataset_train = PadChest_grounding(
    datasetpath=datasetpath,
    split=split, 
    flag_instr=True,
    flag_img=False,
    flag_txt=False
)

datasetpath = os.path.join(DATA_DIR, 'PadChest')
split = 'valid' 
dataset_valid = PadChest_grounding(
    datasetpath=datasetpath,
    split=split, 
    flag_instr=True,
    flag_img=False,
    flag_txt=False
)

prhase_grounding_padchest_dataset = ConcatDataset([dataset_train, dataset_valid])

print("Num samples = " + str(len(prhase_grounding_padchest_dataset)))
print("")



# CONVERSATIONS
print("Conversations standard")

conversation_dir= os.path.join(datasetpath_mimic, 'conversations/train/standard')
filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports')
conv_dataset_standard = MIMIC_Dataset_MM(
    datasetpath=datasetpath_mimic,
    split="train", 
    flag_img=False, 
    flag_instr=False, 
    flag_txt=False, 
    flag_lab=False, 
    filtered_reports_dir=filtered_reports_dir,
    conversation_dir=conversation_dir
    )
print("Num samples = " + str(len(conv_dataset_standard)))
print("")


print("Conversations grounded")

conversation_dir =  os.path.join(datasetpath_mimic, 'conversations/train/grounding') # if present 
filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports')
conv_dataset_grounded = MIMIC_Dataset_MM(
    datasetpath = datasetpath_mimic,
    split="train", flag_img=False, 
    flag_lab=False, only_frontal=True, 
    flag_instr=False, 
    filtered_reports_dir=filtered_reports_dir,
    sentencesBBoxpath = sentencesBBoxpath,
    conversation_dir = conversation_dir,
    classif=False,
    seed=0)

print("Num samples = " + str(len(conv_dataset_grounded)))
print("")

print("Conversations grounded padchest")
datasetpath = os.path.join(DATA_DIR, 'PadChest')
split_train = 'train'
conversation_dir = os.path.join(datasetpath, 'conversations/train/grounding')
dataset_train = PadChest_grounding_per_image(
    datasetpath=datasetpath,
    split=split_train,
    flag_instr=False,
    flag_img=False,
    conversation_dir=conversation_dir
)
# Create the validation dataset.
split_valid = 'valid'
conversation_dir = os.path.join(datasetpath, 'conversations/train/grounding')
# If your valid conversation files are in a different folder, set it accordingly.
dataset_valid = PadChest_grounding_per_image(
    datasetpath=datasetpath,
    split=split_valid,
    flag_instr=False,
    flag_img=False,
    conversation_dir=conversation_dir
)
# Concatenate the two datasets.
conv_dataset_grounded_padchest = ConcatDataset([dataset_train, dataset_valid])

print("Num samples = " + str(len(conv_dataset_grounded_padchest)))
print("")



dataset_info = [
    {
        "dataset":vin_dataset,
        "id_prefix":"vindr-cxr-train1",
    }, 
    {
        "dataset":vin_dataset,
        "id_prefix":"vindr-cxr-train2",
    }, 
    {
        "dataset":vin_dataset_mono,
        "id_prefix":"vindr-cxr-mono-train1",
    }, 
    {
        "dataset":vin_dataset_mono,
        "id_prefix":"vindr-cxr-mono-train1",
    }, 
    {
        "dataset":vin_dataset_mono,
        "id_prefix":"vindr-cxr-mono-train1",
    }, 
    {
        "dataset":prhase_grounding_mscxr_dataset,
        "id_prefix":"mscxr-train1",
    }, 
    {
        "dataset":prhase_grounding_mscxr_dataset,
        "id_prefix":"mscxr-train2",
    }, 
    {
        "dataset":prhase_grounding_mscxr_dataset,
        "id_prefix":"mscxr-train3",
    }, 
    {
        "dataset":prhase_grounding_padchest_dataset,
        "id_prefix":"padchest-train1",
    }, 
    {
        "dataset":prhase_grounding_padchest_dataset,
        "id_prefix":"padchest-train2",
    }, 
    
    {
        "dataset":mimic_dataset_filtered,
        "id_prefix":"mimic-train",
    }, 
    {
        "dataset":chexpertplus_dataset,
        "id_prefix":"chexpertplus-train",
    }, 
    {
        "dataset":chestima_dataset,
        "id_prefix":"chestima-train",
        "num_samples":80000,
    }, 

    {
        "dataset":mimic_dataset_labels,
        "id_prefix":"mimic-labels-train",
    }, 
    {
        "dataset":chexpert_dataset,
        "id_prefix":"chexpert-train",
    }, 
    
    {
        "dataset":conv_dataset_standard,
        "id_prefix":"conv-train",
    }, 
    
    {
        "dataset":conv_dataset_grounded,
        "id_prefix":"conv-grounded-train1",
    }, 
    {
        "dataset":conv_dataset_grounded,
        "id_prefix":"conv-grounded-train2",
    }, 
    
    {
        "dataset":conv_dataset_grounded,
        "id_prefix":"conv-grounded-train3",
    }, 
    {
        "dataset":conv_dataset_grounded,
        "id_prefix":"conv-grounded-train4",
    }, 
    {
        "dataset":conv_dataset_grounded_padchest,
        "id_prefix":"conv-grounded-padchest-train1",
    }, 
    {
        "dataset":conv_dataset_grounded_padchest,
        "id_prefix":"conv-grounded-padchest-train2",
    }, 
    
    {
        "dataset":conv_dataset_grounded_padchest,
        "id_prefix":"conv-grounded-padchest-train3",
    }, 
    {
        "dataset":conv_dataset_grounded_padchest,
        "id_prefix":"conv-grounded-padchest-train4",
    }, 

]

train_llava_dataset = generate_llava_dataset_from_instruction_dataset(dataset_info)
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'llava_datasets', 'all_train.json')
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w") as f:
    json.dump(train_llava_dataset, f, indent=4)

print("LLaVA dataset saved!")

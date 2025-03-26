# RadVLM

## Install dependencies
To install dependencies, execute the following commands:
```
conda create -n radvlm python=3.10 -y
conda activate radvlm
pip install -r requirements.txt
```

## Instruction dataset generation

### Dataset content
The instruction dataset comprises 1,022,742 image-instruction pairs spanning multiple vision-language tasks, including report generation, abnormality classification, anatomical and abnormality grounding, phrase grounding, and conversational interactions. Dataset sources and the corresponding number of image-instruction pairs are listed, with smaller datasets balanced by varying the frequency of instruction occurrences.

| Task                    | Dataset source    | Image-instruction pairs (#) | Evaluation (#) | DUA                                                                                                                                          |
|-------------------------|-------------------|-----------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Report Generation       | MIMIC-CXR         | 230,980 × 1                 | 3,314          | [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
|                         | CheXpert-Plus     | 186,463 × 1                 | -              | [stanfordaimi](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1)                                                   |
| Abnormality classif.    | MIMIC-CXR         | 237,912 × 1                 | 518            | [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
|                         | CheXpert          | 191,027 × 1                 | -              | [stanfordaimi](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)                                                   |
| Anatomical grounding    | Chest Imagenome   | 80,000 × 1                  | 2,000          | [physionet](https://physionet.org/content/chest-imagenome/1.0.0/)                                                                                    |
| Abnormality grounding   | VinDr-CXR         | 16,089 × 3                  | 2,108          | [physionet](https://physionet.org/content/vindr-cxr/1.0.0/)                                                                                         |
| Abnormality detection   | VinDr-CXR         | 15,000 × 2                  | -              | [physionet](https://physionet.org/content/vindr-cxr/1.0.0/)                                                                                         |
| Phrase grounding        | MS-CXR            | 971 × 3                     | 189            | [physionet](https://physionet.org/content/ms-cxr/0.1/)                                                                                              |
|                         | PadChest-GR       | 4,478 × 2                   | -              | [bimcv](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)                                                  |
| Conversation            | MIMIC-CXR         | 86,155 × 1                  | 500            | [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
| Conversation (grounded) | MS-CXR            | 862 × 4                     | 155            | [physionet](https://physionet.org/content/ms-cxr/0.1/)                                                                                              |
|                         | PadChest-GR       | 2,225 × 4                   | -              | [bimcv](https://bimcv.cipf.es/bimcv-projects/padchest) / [bimcv](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)                                                  |

### Datasets download 

Each dataset can be downloaded via the links provided in the right column. Once the access is allowed, the datasets should be organized as follows: 
```
datasets/
├── MIMIC-CXR/
│   ├── mimic-cxr-2.0.0-chexpert.csv
│   ├── mimic-cxr-2.0.0-metadata.csv
│   ├── mimic-cxr-2.0.0-split.csv
│   ├── reports.csv * 
│   ├── files/
│   ├── filtered_reports/ *
│   └── conversations/ *
│   │   ├── train/
│   │   │   ├── standard/
│   │   │   └── grounding/
│   │   └── test/
│   │   │   ├── standard/
│   │   │   └── grounding/
├── CheXpert/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── CheXpertPlus/
│   ├── PNG/
│   │   ├── train/
│   │   └── valid/
│   ├── df_chexpert_plus_240401.csv
│   └── filtered_reports/ * 
├── CHEST_IMA/
│   └── silver_dataset/
├── VinDr-CXR/
│   ├── train_jpg/ * 
│   ├── test_jpg/ * 
│   ├── train/
│   ├── test/
│   ├── annotations_train.csv
│   ├── annotations_test.csv
│   ├── image_resolutions_train.json * 
│   └── image_resolutions_test.json * 
├── MS-CXR/
│   ├── MS_CXR_Local_Alignment_v1.0.0.csv
│   └── sentences_BBox_mscxr/ * 
└── PadChest/
│   ├── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
│   ├── master_table.csv
│   ├── grounded_reports_20240819.json
│   ├── images_grounding/
│   └── conversations/ *
│   │   └── train/
│   │   │   └── grounding/
```

Make sure to set the environment variable `DATA_DIR` to the path of the main datasets directory. For example, if your datasets are located at `/home/username/datasets`, you can set the variable in your shell as follows:
```
export DATA_DIR=/home/username/datasets
```
In the above architecture, the files or folders marked with a `*` were not orginally part of the available datasets, and we describe below the procedure to generate each of them. The rest of the files are directly available in the official repositories. 

## Set up Azure OpenAI
In order to generate synthetic data (see below), you will need to set up environmental variables required to run Azure OpenAI API call. In particular, the following variables should be defined:
```
export AZURE_OPENAI_API_KEY=<your azure openai key>
export AZURE_OPENAI_ENDPOINT=<your azure openai endpoint>
export AZURE_API_VERSION=<your azure openai api version>
```

### Filtering reports in MIMIC-CXR and CheXpert-Plus
- The file `reports.csv` is obtained by following the findings/impression extraction procedure from the [official MIMIC-CXR github](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt). 
- The `filtered_reports` directory contains text reports filtered by the Azure OpenAI API call of GPT-4o. The reports are stored as txt files, organized by `study_id` (e.g., `53862424.txt`). In order to generate this directory, run the following command:
```
python -m radvlm.data.llm_filter_reports --azure_model gpt-4o --split [train,test] --num_chunks [number of parallel API calls] 
```
This command will leverage the GPT-4o prompt stored in `radvlm/data/prefixes_prompts/prefix_filter_reports.txt` to remove statements referring to previous studies. It should be executed for both `train` and `test` split values, in order to construct both `train` and `test` sets. The `--azure_model` parameter is the name of the deployed model on your Azure instance.
Similarly, for CheXpertPlus, we can construct the `filtered_reports` folder, organized by studies, by executing the following command (only for train split):
```
python -m radvlm.data.llm_filter_reports --azure_model gpt-4o --chexpertplus True --split train --num_chunks [number of parallel API calls] 
```

### Converting dicom to jpg in VinDr-CXR
The raw dataset of VinDr-CXR provides images in dicom format in folders `train` and `test`. To obtain the jpg images in directories `train_jpg` and `test_jpg`, as well as the files containing the image dimensions `image_resolutions_train.json` and `image_resolutions_test.json`, execute the following command:
```
python -m radvlm.data.preprocess_scripts.dicom2jpg_vindrcxr
```

### Preprocess grounded phrases in MS-CXR
We re-organize the MS-CXR dataset by creating one json file per image (following MIMIC-CXR `image_id`), with bounding boxes normalized from 0 to 1. These are contained in the directory `sentences_BBox_mscxr/` that can be obtained by executing:
```
python -m radvlm.data.preprocess_scripts.normalize_mscxr
```

### Generate conversations 
For MIMIC-CXR, in order to generate the `conversations` directory, we leverage GPT-4o by providing the corresponding prompt contained in `prefixes_prompts`, and execute the following command:
``` 
python -m radvlm.data.llm_generate_conversations --azure_model gpt-4o --split [train,test] --num_chunks [num API calls]
```
This should be performed for both train and test splits, each containing both standard and grounded conversations (setting the `--grounding` flag). 
For PadChest-GR, set the ` --padchest` flag, and only perform it for the train split and grounding flag. 

### Create final llava dataset 
Once the whole dataset architecture is built, in order to construct the instruction dataset as a unique json file in the llava format, execute the following command:
```
python -m radvlm.data.create_llava_dataset
```
This file contains a list of dictionaries, each following this structure:
```
{
    "image": "path/to/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<question>"
        },
        {
            "from": "gpt",
            "value": "<answer>"
        }
    ],
    "id": "<datapoint-id>"
},
```
where `"image"` refers to the absolute path of the image, `"conversations"` contains the user-assistant instruction (single or multi-turn), and `"id"` is an arbitrary datapoint tag. This structure follows the LLaVA dataset format and can directly be used within their corresponding training script (https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train).

## Finetuning with LLaVA-OneVision

In the `finetuning` directory, we forked from the [official LLaVA-OneVision repo](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main) and adapted for the case of finetuning the RadVLM model on CXR data.

### Create finetuning environment 
Install the packages that are specific to LLaVA-OneVision repository.
```
conda create -n llava python=3.10 -y
conda activate llava
cd finetuning
pip install --upgrade pip
pip install -e ".[train]"
```
### Training
The training script `finetune_radio_7b.sh` is provided in the `script` folder. It is adapted to train a base [llava-onevision checkpoint](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-si) on the curated Instruction dataset of RadVLM from the previous steps (`all_train.json`).
The training script accesses this dataset via the argument `data_path`, hyperparameters such as learning rate or number of epochs can be modified at convenience, as well as the training starting point that could be an already trained checkpoint. 

## Evaluation 

For the evaluation, activate the `radvlm` environment previously created:
```
conda activate radvlm
```

### Conversion of llava-ov checkpoint to huggingface format 
A first step consists of converting the RadVLM checkpoint obtained after finetuning llava-onevision on the radiology instruction dataset, following the finetuning section. In the case of a 7B checkpoint, this can be performed by executing the following command: 
```
python -m radvlm.evaluation.convert_llava_onevision_weights_to_hf --model_id lmms-lab/llava-onevision-qwen2-7b-si --model_path $CKPT_PATH_RADVLM
```
The converted HF model will be stored in the same directory as the finetuned checkpoint, with the additional `_hf` suffixe. 

### Baseline models implementation 
Baseline models used in the paper to compare performance metrics are re-implemented within this repo, and their corresponding loading and inference scripts are stored in the file `models_loading_inference.py`. For the specific case of RaDialog, an additional command should be executed inside the evaluation directory: 
```
git clone https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation
```

### Model evaluation on single instructions
All instruction tasks (report generation, abnormality classification, visual grounding) are evaluated on the test sets of the dataloaders provided in the `data` repo. In order to evaluate a specific model (RadVLM or baseline model), execute this command (scaling to number of available GPUs): 
```
accelerate launch --num_processes=4 radvlm.evaluation.evaluate_instructions --task [report_generation, abnormality_classification, region_grounding, abnormality_grounding]  --model_name [radialog, llavamed, chexagent, maira2, llavaov, $CKPT_PATH_RADVLM] 
```
The tasks that can be evaluated for each model is summarized in the following table:

| Model          | Report | Classification | Grounding | Conversation |
|----------------|:------:|:--------------:|:---------:|:------------:|
| LLaVA-OV       |   ✔    |       ✘        |     ✘     |      ✔       |
| LLaVA-Med      |   ✔    |       ✘        |     ✘     |      ✔       |
| RaDialog       |   ✔    |       ✔        |     ✘     |      ✔       |
| CheXagent      |   ✔    |       ✔        |     ✔     |      ✘       |
| MAIRA-2        |   ✔    |       ✘        |     ✔     |      ✘       |
| **RadVLM**     |   ✔    |       ✔        |     ✔     |      ✔       |

To evaluate generated reports with the GREEN metric, after the above command is executed for the `report_generation` task, run the following command:
```
torchrun --nproc_per_node=4 -m radvlm.evaluation.eval_green --model_name [radialog,llavamed, chexagent, maira2, llavaov, $CKPT_PATH_RADVLM]
```

### Model evaluation for multi-round conversations
To evaluate a model on the test set of multi-round conversation tasks, execute the following command:
```
python -m radvlm.evaluation.evaluate_conversations --azure_model gpt-4o --model_name [radialog, llavamed, $CKPT_PATH_RADVLM] 
```
This will evaluate the model over the questions of the test set of the conversation dataset, by comparing with the ground truth to expected answers. An average score is cumulatively computed over the test dataset iterations. In order to evaluate on the grounded dataset, set the `--grounding` flag.







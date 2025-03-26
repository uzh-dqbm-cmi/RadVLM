import csv
import json
import os
from collections import defaultdict

DATA_DIR = os.environ.get('DATA_DIR')
if DATA_DIR is None:
    raise EnvironmentError("The environment variable 'DATA_DIR' is not set.")

mscxr_data_dir = os.path.join(DATA_DIR, 'MS-CXR')

# Input CSV file
csv_file = os.path.join(mscxr_data_dir, 'MS_CXR_Local_Alignment_v1.0.0.csv')

# Output directory
output_dir = os.path.join(mscxr_data_dir, 'sentences_BBox_mscxr')
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store data by dicom_id and label_text
# data_by_id = {
#   dicom_id: {
#       label_text: [(x, y, w, h, image_width, image_height), ...],
#       ...
#   },
#   ...
# }

data_by_id = {}

# Read the CSV
with open(csv_file, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dicom_id = row['dicom_id']
        label_text = row['label_text']
        x = float(row['x'])
        y = float(row['y'])
        w = float(row['w'])
        h = float(row['h'])
        image_width = float(row['image_width'])
        image_height = float(row['image_height'])

        if dicom_id not in data_by_id:
            data_by_id[dicom_id] = defaultdict(list)
        data_by_id[dicom_id][label_text].append((x, y, w, h, image_width, image_height))

# Process each dicom_id and write to JSON
for dicom_id, obs_dict in data_by_id.items():
    result_list = []

    for label_text, boxes in obs_dict.items():
        for (x, y, w, h, img_w, img_h) in boxes:
            # Convert to normalized coordinates
            xmin = x / img_w
            ymin = y / img_h
            xmax = (x + w) / img_w
            ymax = (y + h) / img_h

            obs_entry = {
                "observation": label_text,
                "box": [xmin, ymin, xmax, ymax]
            }
            result_list.append(obs_entry)

    # Write the json file named [dicom_id].json
    output_path = os.path.join(output_dir, f"{dicom_id}.json")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(result_list, outfile, indent=4, ensure_ascii=False)

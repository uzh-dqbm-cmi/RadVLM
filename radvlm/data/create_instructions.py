import random
import numpy as np
from collections import defaultdict
from collections import Counter
from torch.utils.data import DataLoader
from radvlm.data.utils import custom_collate_fn


def format_boxes(bounding_boxes, num_float=2):
    formatted_boxes = [
        f"[{round(bbox[0], num_float)}, {round(bbox[1], num_float)}, {round(bbox[2], num_float)}, {round(bbox[3], num_float)}]"
        for bbox in bounding_boxes
    ]
    boxes_str = (
        ", ".join(formatted_boxes[:-1])
        + ("" if len(formatted_boxes) < 2 else " and ")
        + formatted_boxes[-1]
    )
    return boxes_str


def select_article(word):
    if word[0].lower() in "aeiou":
        return "an"
    else:
        return "a"
    
    
def create_json_cell_llava(sample, id_prefix, sample_idx, dataset):
    sample_instr = sample["conversation"] if "conversation" in sample else sample["instr"]
    json_cell = {
        "image": sample["img_path"],
        "conversations": [],
        "id": f"{id_prefix}_{sample_idx}"
    }

    if isinstance(sample_instr, dict):
        # Case when sample_instr is a dict with question and answer
        sample_instr = [sample_instr]  # Convert to a list for uniform processing
    

    for j, instr in enumerate(sample_instr):
        if "from" in instr and "value" in instr:
            # Case when it's already in the correct format
            conv_cell = instr.copy()
            if j == 0:
                conv_cell["value"] = f"<image>\n{conv_cell['value']}"
            json_cell["conversations"].append(conv_cell)
        elif "question" in instr and "answer" in instr:
            # Case when it's a dict with question and answer keys
            conv_cell_human = {
                "from": "human",
                "value": instr["question"]
            }
            if j == 0:
                conv_cell_human["value"] = f"<image>\n{conv_cell_human['value']}"
            conv_cell_ai = {
                "from": "gpt",
                "value": instr["answer"]
            }
            json_cell["conversations"].append(conv_cell_human)
            json_cell["conversations"].append(conv_cell_ai)
        else:
            continue

    if "labels" in sample:
        json_cell["labels"] = sample["labels"]
    if getattr(dataset, "pathologies", None):
        json_cell["pathologies"] = dataset.pathologies.copy()

    return json_cell


    
def generate_llava_dataset_from_instruction_dataset(dataset_info, batch_size=64, num_workers=128, seed=0):
    """
    Generate structured data that will be the source of a Llava dataset (json) based on the Dataset.
    The dataset provided to this function must be a dataset that returns a dictionary with "instr" and "img_path" as keys.
    Without the "num_samples", the function will use all the samples in the dataset.
    """
    json_structure = []

    for dataset_i, dataset_info_cell in enumerate(dataset_info):
        # Define DataLoader
        dataset = dataset_info_cell["dataset"]
        id_prefix = dataset_info_cell["id_prefix"]
        print(f"Processing {id_prefix}")
        np.random.seed(seed)
        random.seed(seed)
        if "id_prefix" not in dataset_info_cell:
            dataset_info_cell["id_prefix"] = dataset_i
        
        num_samples = dataset_info_cell.get("num_samples", len(dataset))
        
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,  # You can adjust the batch size
            shuffle=True,  # Shuffle dataset
            num_workers=num_workers,  # High number of workers for parallel data loading
            pin_memory=True,  # Use pinned memory to speed up transfers to GPU
            collate_fn=custom_collate_fn  # Custom collate function to handle None values
        )

        sample_count = 0
        for batch in data_loader:
            for sample in batch:
                if sample_count >= num_samples:
                    break
                json_cell = create_json_cell_llava(sample, dataset_info_cell["id_prefix"], len(json_structure), dataset)
                json_structure.append(json_cell)
                sample_count += 1

            if sample_count >= num_samples:
                break
                
    return json_structure

import random

def generate_instruction_report_generation(text, german_suffixe=False):
    """
    Generate a 1-round instruction (question and answer) for creating a radiology report
    from an X-ray/image, with a preference for selecting 'no preamble' in the answer most of the time.

    Parameters:
    - text (str): The radiology report text to include in the answer.
    - german (bool): If True, use German questions and preambles.
    - german_suffixe (bool): If True, add 'in German' to questions.

    Returns:
    - A dictionary containing 'question' and 'answer' keys.
    """

    # Possible question variations for the user to write a radiology report
    questions = [
        "Provide a radiology report for this Chest X-Ray.",
        "Analyze the Chest X-Ray and report your findings.",
        "Please give a detailed radiology report from this Chest X-Ray image.",
        "Please offer a detailed analysis based on this image.",
        "Analyze this image and provide a report.",
        "Write a radiology report for this Chest X-ray.",
        "Draft a concise report for this image.",
        "Please provide a brief radiology report.",
        "Compose a radiology report focusing on key findings.",
        "Create a structured radiology report for this X-ray.",
        "Produce a summary radiology report for this chest image.",
        "Provide a radiology report for this exam.",
        "Write a report summarizing this Chest X-ray.",
        "Draft a simple radiology report for these findings.",
        "Outline a radiology report for this X-ray."
    ]


    # Add "in German" to questions if german_suffixe is True
    if german_suffixe:
        questions = [q + " in German" for q in questions]

    # Select a random question and preamble
    question = random.choice(questions)

    # Format the instruction
    instruction = {"question": question, "answer": f"{text}"}

    return instruction


def generate_instruction_phrase_location(bounding_boxes, label):

    question_variations = [
        "Please locate the following sentence: {}",
        "Identify the position of the following phrase in the CXR: {}",
        "Please show me the location of: {}",
        "Highlight the area of the following observation on the image: {}",
        "Where on the image can you see the following observation: {}",
        "Find the region corresponding to: {}",
        "Please indicate where this finding is located: {}",
        "Mark the area where you observe: {}"
    ]

    answer_variations = [
        "This sentence is located at the coordinates {} on the image.",
        "You'll find it at {} in the CXR.",
        "This phrase can be observed at {} on the image.",
        "The bounding box for this observation is {}.",
        "Its location is given by {}.",
        "It is displayed at {} in the radiograph.",
        "The area specified is at coordinates {}.",
        "This finding is located at {} in the image."
    ]
    

    boxes_str = format_boxes(bounding_boxes)


    # Randomly pick a question and answer variation
    if label[0].isupper() and not label.isupper():
        label = label.lower()
    question = random.choice(question_variations).format(label)
    answer = random.choice(answer_variations).format(boxes_str)

    instruction = {"question": question, "answer": answer}

    return instruction




def generate_instruction_location(bounding_boxes, label):
    """
    Generate a question and answer pair from a set of predefined variations,
    about the location of a random anatomical region on a Chest X-ray or image,
    based on provided bounding boxes and region labels.

    Parameters:
    - bounding_boxes (list of lists): A list where each element is a bounding box
      represented as [x1, y1, x2, y2], assumed to be integer values that will be normalized.
    - regions (list of str): A list of region names corresponding to each bounding box.

    Returns:
    - A JSON string containing a dict for the question and the answer.
    """


    # Define 20 variations of questions and answers
    questions_variations = [
        "Where is the {} located in this Chest X-ray?",
        "Can you point out the {}'s position on the image?",
        "What's the location of the {} in the X-ray?",
        "Identify where the {} is on this Chest X-ray, please.",
        "Where exactly is the {} found on this image?",
        "Could you specify where to find the {} on this X-ray?",
        "Highlight the {}'s area on the image.",
        "Show me the {}'s location on this CXR.",
        "Where should I look to find the {} in this image?",
        "Can you locate the {} on this X-ray for me?",
        "Please point to the {} on this Chest X-ray.",
        "Indicate the position of the {} on this image.",
        "Describe the location of the {} on the X-ray.",
        "Where on this image is the {} located?",
        "Point out the exact location of the {} in the Chest X-ray.",
        "How can I identify the {} on this image?",
        "Where is the {} situated in this CXR?",
        "Can you highlight the {} on this image?",
        "Indicate where the {} is found on this X-ray.",
        "Describe where to find the {} on this Chest X-ray.",
    ]

    answer_variations = [
        "The {} is located at the coordinates {} on the image.",
        "You'll find the {} at {} in the X-ray.",
        "The {} can be seen at {} on the Chest X-ray.",
        "The location of the {} is at {} on the image.",
        "For the {}, the coordinates are {} on the X-ray.",
        "The {} is situated at {} in the image.",
        "On the Chest X-ray, the {} is located at {}.",
        "The {} appears at the coordinates {} on the image.",
        "In the X-ray, the {} is identifiable at {}.",
        "The location for the {} is marked at {} on the Chest X-ray.",
        "The {} is positioned at {} on the image.",
        "The area occupied by the {} is at {} in the X-ray.",
        "On the image, you can find the {} at {}.",
        "The {}'s  location is at {} on the Chest X-ray.",
        "In terms of coordinates, the {} is found at {} on the image.",
        "Regarding the {}, it is located at {} on the X-ray.",
        "The {} specifically is at {} on the Chest X-ray.",
        "Concerning the {}, you will find it at {} in the image.",
        "The {} is at {} on the X-ray.",
        "For identifying the {}, look at {} on the Chest X-ray.",
    ]


    # Format the coordinates as normalized floats with two or three decimal places
    boxes_str = format_boxes(bounding_boxes)


    # Randomly pick a question and answer variation
    if label[0].isupper() and not label.isupper():
        label = label.lower()
    question = random.choice(questions_variations).format(label)
    answer = random.choice(answer_variations).format(label, boxes_str)

    instruction = {"question": question, "answer": answer}

    return instruction



def generate_instruction_abnormalities_grouped(bounding_boxes, abnormalities):
    """
    Generate a question and an answer about the presence and location of abnormalities
    (or lesions) on a Chest X-ray, incorporating variations for questions and answers and
    correctly using articles ('a' or 'an') before abnormalities. Abnormalities mentioned multiple
    times will be grouped with all their associated bounding boxes.

    Parameters:
    - bounding_boxes (list of lists): A list where each element is a bounding box
      represented as [x1, y1, x2, y2], assumed to be integer values.
    - abnormalities (list of str): A list of abnormality names corresponding to each bounding box.

    Returns:
    - A JSON string containing a dict for the question and the answer.
    """

    question_variations = [
        "Could you indicate if there are any abnormalities on this Chest X-ray and their locations?",
        "Are abnormalities present on this Chest X-ray? Where exactly can they be found?",
        "Please identify any lesions or abnormalities on this X-ray and specify their locations.",
        "On this Chest X-ray, can you point out any abnormalities and their precise positions?",
        "I need information on any abnormalities or lesions on this X-ray, including their locations. Can you help?",
        "Can you detect and describe the location of any abnormalities found on this Chest X-ray?",
        "Are there identifiable abnormalities on this Chest X-ray? If so, where are they located?",
        "Tell me about any lesions on this X-ray and detail their specific locations.",
        "Do any abnormalities appear on this Chest X-ray? Please point them out along with their locations.",
        "Identify any abnormalities or lesions present on this X-ray and provide their exact locations.",
    ]

    answer_prefix_variations = [
        "Sure! I can find",
        "Indeed, there are",
        "Yes, the following abnormalities are identified:",
        "Upon examination, I detect",
        "The analysis reveals",
        "After a detailed review, we have discovered",
        "The findings include",
        "Notably, I can identify",
        "Based on the image, there are",
        "From the examination, it's evident there are",
    ]

    # Select a random question variation
    question = random.choice(question_variations)

    no_lesions_answers = [
        "I can't find any lesion on the image.",
        "No abnormalities or lesions are detected on this Chest X-ray.",
        "The Chest X-ray appears to be clear of any lesions or abnormalities.",
        "Upon review, no lesions are visible on the image.",
        "The examination reveals a clean bill of health with no visible abnormalities.",
        "This Chest X-ray shows no signs of abnormalities or lesions.",
        "No lesions or abnormalities are present on this X-ray, as far as I can tell.",
        "After a thorough examination, I conclude that there are no detectable lesions on this X-ray.",
        "The image does not display any abnormalities or lesions.",
        "Based on this X-ray, it appears there are no lesions or abnormalities to report.",
    ]

    if not bounding_boxes or not abnormalities:
        answer = random.choice(no_lesions_answers)
    else:
        if len(bounding_boxes) != len(abnormalities):
            raise ValueError(
                "Bounding boxes and abnormalities lists must be of equal length."
            )

        # Assuming abnormalities and bounding_boxes are already defined lists
        abnormalities_dict = defaultdict(list)
        for abnormality, bbox in zip(abnormalities, bounding_boxes):
            abnormalities_dict[abnormality].append(bbox)

        # Format each abnormality with its associated coordinates and the correct article
        abnormalities_descriptions = []
        for abnormality, boxes in abnormalities_dict.items():
            article = select_article(abnormality)
            abnormality = abnormality.lower()
            boxes_str = format_boxes(boxes)
            abnormalities_descriptions.append(
                f"{article} {abnormality} located at the coordinates {boxes_str}"
            )

        # Join all abnormalities descriptions
        abnormalities_str = "; ".join(abnormalities_descriptions)

        # Select a random answer prefix variation
        answer_prefix = random.choice(answer_prefix_variations)
        answer = f"{answer_prefix} {abnormalities_str}."

    instruction = {"question": question, "answer": answer}

    return instruction


def generate_instruction_foreign_objects(bounding_boxes):
    """
    Generate a question and an answer about the location of potential foreign objects
    on a Chest X-ray, incorporating variations for questions and answers.

    Parameters:
    - bounding_boxes (list of lists): A list where each element is a bounding box
      represented as [x1, y1, x2, y2], assumed to be integer values.

    Returns:
    - A JSON string containing a dict for the question and the answer.
    """

    question_variations = [
        "Could you indicate if there are any foreign objects on this Chest X-ray and their locations?",
        "Are there any foreign objects visible on this Chest X-ray and, if so, where?",
        "Can you detect and pinpoint the location of any foreign objects on this X-ray?",
        "Do you see any foreign objects present on this Chest X-ray? Where exactly?",
        "Are any unexpected items or foreign objects showing up in this X-ray image?",
        "Can you identify if there are foreign objects in this X-ray, and provide their locations?",
        "Could you locate and describe any foreign objects in this Chest X-ray?",
        "On this X-ray, are there signs of any foreign objects? Where can they be found?",
        "Please point out any foreign objects on this X-ray along with their coordinates.",
        "Can you report the presence and locations of any foreign objects in this Chest X-ray?",
    ]

    answer_prefix_variations = [
        "Upon review, I have identified foreign objects located at the following coordinates:",
        "The X-ray reveals foreign objects at:",
        "Foreign objects are detected at these positions:",
        "I've located potential foreign objects in the X-ray at:",
        "The analysis indicates the presence of foreign objects at:",
        "Foreign objects are visible in the X-ray at:",
        "The following foreign objects have been found at:",
        "Detectable foreign objects are present at:",
        "The image analysis shows foreign objects located at:",
        "Foreign objects have been pinpointed at the following coordinates:",
    ]

    no_objects_answers = [
        "No foreign objects are detected on this Chest X-ray.",
        "The Chest X-ray is clear of any foreign objects.",
        "There are no detectable foreign objects in this X-ray image.",
        "I find no evidence of foreign objects on this X-ray.",
        "The X-ray does not show any foreign objects.",
        "No unexpected items or foreign objects are visible in the X-ray.",
        "There are no signs of foreign objects on this X-ray.",
        "The X-ray appears clean, with no foreign objects present.",
        "No foreign objects can be seen in the X-ray image.",
        "There's nothing that resembles foreign objects in this X-ray.",
    ]

    # Select a random question variation
    question = random.choice(question_variations)

    if len(bounding_boxes)==0:
        answer = random.choice(no_objects_answers)
    else:
        boxes_str = format_boxes(bounding_boxes)
        # Select a random answer prefix variation
        answer_prefix = random.choice(answer_prefix_variations)
        answer = f"{answer_prefix} {boxes_str}."

    instruction = {"question": question, "answer": answer}

    return instruction





def generate_instruction_abnormalities(abnormalities):
    """
    Generate a question and an answer about the presence of abnormalities
    (or lesions) on a Chest X-ray, incorporating variations for questions and answers.
    Abnormalities mentioned multiple times will be grouped together.

    Parameters:
    - abnormalities (list of str): A list of abnormality names.

    Returns:
    - A JSON string containing a dict for the question and the answer.
    """

    question_variations = [
        "Can you tell me if there are any abnormalities on this image?",
        "Are there any abnormalities on this Chest X-ray?",
        "Please identify abnormalities on this X-ray.",
        "Can you point out if there are any abnormalities on this Chest X-ray?",
        "Are there abnormalities on this Chest X-ray?",
        "Tell me about abnormalities on this image",
        "Do any abnormalities appear on this Chest X-ray?",
        "Identify abnormalities on this image",
        "Could you indicate if there are any lesions on this Chest X-ray?",
        "Are there any abnormalities present on this image?",
        "Please identify the abnormalities from this image.",
    ]

    if not abnormalities:
        no_abnormalities_answers = [
            "No abnormalities or lesions are detected on this Chest X-ray.",
            "The Chest X-ray appears to be clear of any abnormalities.",
            "There are no detectable abnormalities on this X-ray.",
            "This Chest X-ray shows no signs of abnormalities.",
            "Based on this X-ray, there are no abnormalities to report.",
            "There are no findings on this Chest X-ray.",
            "The Chest X-ray appears to be clear of any findings.",
        ]
        answer = random.choice(no_abnormalities_answers)
    else:
        # Count occurrences of each abnormality and group them
        abnormality_count = Counter(abnormalities)
        abnormalities_list = [
            f"{abnormality.lower()}" for abnormality, _ in abnormality_count.items()
        ]

        # Join all abnormalities descriptions
        if len(abnormalities_list) > 1:
            abnormalities_str = (
                ", ".join(abnormalities_list[:-1]) + " and " + abnormalities_list[-1]
            )
            answer_variations = [
                "The following abnormalities are identified:",
                "The following abnormalities are present:",
                "The analysis reveals these abnormalities:",
                "From the examination, it is evident that these abnormalities are present:",
                "The Chest X-ray includes the following abnormalities:",
            ]
        else:
            # There is only one abnormality
            abnormalities_str = abnormalities_list[0]
            answer_variations = [
                "Yes, there is one abnormality identified:",
                "The following abnormality is present:",
                "The analysis reveals the following abnormality:",
                "The Chest X-ray includes the following finding:",
                "Yes, there is one abnormality identified:",
            ]

        # Select a random answer prefix variation
        answer_prefix = random.choice(answer_variations)
        answer = f"{answer_prefix} {abnormalities_str}."

    # Select a random question variation
    question = random.choice(question_variations)

    instruction = {"question": question, "answer": answer}

    return instruction

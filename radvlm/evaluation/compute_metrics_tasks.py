from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import re
from radvlm.evaluation.vilmedic.utils import calcAllMetrics_whole

def evaluate_results(task, output, dataset):
    """
    Evaluate the results based on the task.

    Args:
        task (str): The task performed.
        output (list): The inference output.
        dataset (Dataset): The dataset used.

    Returns:
        dict: The evaluation metrics.
    """
    if task in ["object_grounding", "region_grounding", "abnormality_grounding", "abnormality_detection", "phrase_grounding"]:
        metrics = evaluate_boxes(output, avg_iou=True)

    elif task == "abnormality_classification":
        labels = [element.lower() for element in dataset.pathologies] 
        metrics = evaluate_classification(output, labels)

    elif task == "report_generation":
        list_predictions = [item["output"] for item in output]
        list_groundtruth = [item["txt"] for item in output]
        metrics = evaluate_reports(list_groundtruth, list_predictions)

    else:
        raise ValueError(f"Unsupported task: {task}")
    
    for key, value in metrics.items():
        print(f"{key}: {round(float(value)*100, 1)}")

    return metrics



def evaluate_reports(output_report_list, gt_report_list):
    """
    Evaluate the model's predicted reports against ground truth reports.
    output_report_list (list): List of model outputs(String).
    gt_report_list (list): List of ground truth reports(String).
    Returns:
        dict: A dictionary containing performance metrics (BLEU, ROUGE, METEOR, CIDEr).

        Example:
        {'blue': 0.03780220314351944, 'bertscore': 0.evaluate_classification6388142704963684, 'meteor': 0.21547559264892072,
         'ciderd': 0.6235056291023062, 'rouge1': 0.4504924864407623, 'rouge2': 0.14196560646771025,
         'rougel': 0.38260306264616606, 'radgraph_simple': 0.16935286935286936,
         'radgraph_partial': 0.12768620268620268, 'radgraph_complete': 0.10158730158730159,
         'chexbert_all_micro': 0.8148148148148148, 'chexbert_all_macro': 0.5476190476190476,
         'chexbert_5_micro': 0.9230769230769231, 'chexbert_5_macro': 0.9333333333333332}
    """
    metrics = calcAllMetrics_whole(output_report_list, gt_report_list)

    return metrics


def extract_bounding_boxes(answer):
    """Extract bounding boxes from the answer string."""
    pattern = r"\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"
    return [list(map(float, match)) for match in re.findall(pattern, answer)]

def compute_iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def compute_average_precision(recall, precision):
    """Compute Average Precision (AP) given precision and recall values."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # Ensure precision is monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def evaluate_boxes(output_list, iou_thresholds=None, avg_iou=False):
    """Evaluate the model's predicted bounding boxes against ground truth boxes using a custom method."""
    if iou_thresholds is None:
        iou_thresholds = [0.5]  # Default to 0.5 if no thresholds are provided
    
    results = {}
    total_iou_sum = 0
    total_boxes_count = 0

    for iou_threshold in iou_thresholds:
        all_precisions = []
        all_recalls = []

        for output_single in output_list:
            if not ("output" in output_single and "boxes" in output_single):
                raise ValueError("Both keys 'output' and 'boxes' must be contained in dict.")

            output_text = output_single["output"]
            predicted_boxes = extract_bounding_boxes(output_text)
            actual_boxes = output_single["boxes"]

            if len(predicted_boxes) == 0 or len(actual_boxes) == 0:
                all_precisions.append(0)
                all_recalls.append(0)
                continue

            ious = np.zeros((len(predicted_boxes), len(actual_boxes)))

            # Compute IoU between each pair of predicted and ground truth boxes
            for i, pred_box in enumerate(predicted_boxes):
                for j, gt_box in enumerate(actual_boxes):
                    ious[i, j] = compute_iou(pred_box, gt_box)

            # If avg_iou is True, calculate the average IoU across all matches
            if avg_iou:
                total_iou_sum += np.sum(ious)
                total_boxes_count += len(predicted_boxes) * len(actual_boxes)

            # Match predicted boxes to ground truth boxes
            matched_gt = set()
            true_positives = np.zeros(len(predicted_boxes))
            false_positives = np.zeros(len(predicted_boxes))

            for i, pred_box in enumerate(predicted_boxes):
                max_iou_idx = np.argmax(ious[i, :])
                max_iou = ious[i, max_iou_idx]

                if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
                    true_positives[i] = 1
                    matched_gt.add(max_iou_idx)
                else:
                    false_positives[i] = 1

            tp_cumsum = np.cumsum(true_positives)
            fp_cumsum = np.cumsum(false_positives)
            recall = tp_cumsum / len(actual_boxes)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)

            # Compute average precision for this sample
            ap = compute_average_precision(recall, precision)
            all_precisions.append(ap)
            all_recalls.append(np.max(recall))

        # Compute mean average precision (mAP)
        mAP = np.mean(all_precisions) if all_precisions else 0.0
        results[f"mAP_{iou_threshold}"] = mAP

    # Calculate and add average IoU to results if avg_iou is True
    if avg_iou and total_boxes_count > 0:
        avg_iou_value = total_iou_sum / total_boxes_count
        results["avg_iou"] = avg_iou_value

    return results



def evaluate_classification(output_list, labels):
    """Process the data to match classification with output and calculate metrics.
    Args:
        output_list (list of dicts): List of dicts containing model outputs and actual labels.
        labels (list): List of all possible labels.
    Returns:
        dict: A dictionary containing processed data and performance metrics (Accuracy, F1 Score).
    """
    ret = []
    predicted = []
    actual = []

    # Create a mapping of labels to indices for compatibility with sklearn functions
    label_to_index = {label.lower(): idx for idx, label in enumerate(labels)}

    for output_single in output_list:
        if not ("output" in output_single and "labels" in output_single):
            raise ValueError("Both keys 'output' and 'labels' must be contained in dict.")

        ret_cell = output_single.copy()
        output_text = output_single["output"].lower()
        predicted_labels = [label for label in labels if label.lower() in output_text]
        actual_labels = [label.lower() for label in output_single["labels"]]

        correct_labels = [label for label in predicted_labels if label in actual_labels]
        incorrect_labels = [label for label in predicted_labels if label not in actual_labels]
        missing_labels = [label for label in actual_labels if label not in predicted_labels]
        predicted_arrays = [1 if label.lower() in predicted_labels else 0 for label in labels]
        actual_arrays = [1 if label.lower() in actual_labels else 0 for label in labels]

        predicted.append(predicted_arrays)
        actual.append(actual_arrays)
        ret_cell["correct_labels"] = correct_labels
        ret_cell["incorrect_labels"] = incorrect_labels
        ret_cell["missing_labels"] = missing_labels
        ret.append(ret_cell)

    predicted = np.array(predicted)
    actual = np.array(actual)
    
    # Calculate macro and micro metrics
    precision_micro, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(actual, predicted, average='micro')
    precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(actual, predicted, average='macro')

    # Calculate F1 score per label
    _, _, f1_scores_per_label, _ = precision_recall_fscore_support(actual, predicted, average=None)

    # Organize per-label F1 scores using original label names
    per_label_f1_scores = {label: f1_score for label, f1_score in zip(labels, f1_scores_per_label)}

    # Metrics dictionary
    metrics = {
        "Precision(macro)": precision_macro,
        "Precision(micro)": precision_micro,
        "Recall(macro)": recall_macro,
        "Recall(micro)": recall_micro,
        "F1 Score(macro)": f1_score_macro,
        "F1 Score(micro)": f1_score_micro,
        **per_label_f1_scores
    }
    
    return metrics
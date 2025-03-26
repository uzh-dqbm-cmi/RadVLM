import json
from green_score import GREEN
import os 
import argparse

parser = argparse.ArgumentParser(description="A script to evaluate reports with the GREEN metric.")
parser.add_argument('--model_name', type=str, default='radialog', help="The VLM to evaluate")
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, 'results', args.model_name + '_report_generation_output.json')

# Open and read the JSON file from the 'results' directory
with open(results_path, 'r') as file:
    output = json.load(file)

list_predictions = [item["output"] for item in output]
list_groundtruth = [item["txt"] for item in output]

model_name = "StanfordAIMI/GREEN-radllama2-7b"

green_scorer = GREEN(model_name, output_dir=".")
mean, std, green_score_list, summary, result_df = green_scorer(list_groundtruth, list_predictions)

print(mean)
print(summary)

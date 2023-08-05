### predict result for task1 by using RoBERTa


import argparse
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='predict result for task1 by using RoBERTa')

    parser.add_argument('--input', type=str, help='input data path', required=True)
    parser.add_argument('--output', type=str, help='output data path', required=False)

    return parser.parse_args()


def load_input(df):
    with open(df, 'r') as inp:
         inp = [json.loads(i) for i in inp]
    return pd.DataFrame(inp)


class ClfModel:
    def __init__(self):
        self.models = {
            "passage": AutoModelForSequenceClassification.from_pretrained("./models/roberta-base-3_passage", num_labels=2,local_files_only=True),
            "phrase": AutoModelForSequenceClassification.from_pretrained("./models/roberta-base-2_phrase", num_labels=2,local_files_only=True),
            "multi": AutoModelForSequenceClassification.from_pretrained("./models/roberta-base-3_multi", num_labels=2,local_files_only=True)
        }
        self.tokenizer = AutoTokenizer.from_pretrained("./models/roberta-base-3_multi",local_files_only=True)
        self.fields = {"passage": ['postText', 'targetTitle'], "phrase": ['postText'], "multi": ['postText', 'targetTitle', 'targetParagraphs']}

    def get_text(self, row, tag):
        text = ""
        for field in self.fields[tag]:
            if isinstance(row[field], list):
                text += ' '.join(row[field])
            elif isinstance(field, str):
                text += row[field]
            else:
                raise NotImplemented
        return text

    def predict_probability(self, text: str, model):
        tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**tokenized).logits
        return logits.argmax()

    def predict(self, row: str):
        probabilities = {}
        for tag_name, model in self.models.items():
            text = self.get_text(row, tag_name)
            probability = self.predict_probability(text, model)
            probabilities[tag_name] = probability

        return max(probabilities, key=probabilities.get)


def predict(file_path):
    df = load_input(file_path)
    uuids = list(df['id'])

    classifyer = ClfModel()

    for idx, i in tqdm(df.iterrows()):
        spoiler_type = classifyer.predict(i)
        yield {'id': uuids[idx], 'spoilerType': spoiler_type}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    print(args.input)
    run_baseline(args.input, args.output)
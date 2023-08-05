### Roberta

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DataCollatorWithPadding
#import wandb
from datetime import datetime
import os

def DataLoader(fields: List[str], files: Dict[str, str], mapping: Dict[str, int]) -> pd.DataFrame:
    dataset = {}

    def encode_label(label: str):
        return mapping[label]

    def load_data(file: str):
        df = pd.read_json(file, lines=True)

        data = []
        for _, i in df.iterrows():
            text = ""
            for field in fields:
                if isinstance(i[field], list):
                    text += ' '.join(i[field])
                elif isinstance(field, str):
                    text += i[field]
                else:
                    raise NotImplemented

            data.append({
                "text": text,
                "label": encode_label(i["tags"][0])})
        return data


    for split in list(files.keys()):
        dataset[split] = load_data(files[split])

    return dataset

def preprocess(dataset: List[Dict[str, List[str]]], model_base: str, model_name: str):
    if model_base.startswith("roberta"):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

    elif model_base.startswith("bert"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        raise NotImplementedError

    # create tokanizer function with correct tokenizer
    def preprocess_function(sample):
        return tokenizer(sample["text"], truncation=True)

    # process slices
    dataset_tokenized = {}
    for slice in list(dataset.keys()):
        slice_dataset = Dataset.from_list(dataset[slice])
        slice_tokenized = slice_dataset.map(preprocess_function, batched=True)
        dataset_tokenized[slice] = slice_tokenized

    return dataset_tokenized

def create_model(model_base: str, model_name: str, file_name: str, dataset):
    # Get num labels
    num_label = len(dataset[list(dataset.keys())[0]].unique("label"))

    if model_base.startswith("roberta"):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

    elif model_base.startswith("bert"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        raise NotImplementedError

    # Metrics
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    # Model
    training_args = TrainingArguments(
        output_dir="./checkpoints"+file_name,
        logging_dir='./LMlogs',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps = 10,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        evaluation_strategy='epoch',
        save_strategy='epoch',)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer

def config_name(model, fields, mapping):
    if sum(mapping.values())> 1:
        type_ = "multiclass"
    else:
        type_ =  [k for k, v in mapping.items() if v == 1][0]
    return model+"-"+str(len(fields))+"_"+type_

if __name__ == '__main__':

    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("GPU is not available.")

    # Grid search parameter
    models = ["roberta-base"]
    field_config = [["postText", "targetTitle", "targetParagraphs"]]
    types = ["multiclass"]

    # get confogurations
    configs = []
    for model in models:
        for fields in field_config:
            for type in types:
                if type == "multiclass":
                    mapping = {'passage': 0, 'phrase': 1, 'multi': 2}
                    configs.append((model, fields, mapping))


                elif type == "one_against_the_others":
                    classes = ["passage", "phrase", "multi"]
                    for class_ in classes:
                        mapping = {}
                        for c in classes:
                            if c == class_:
                                mapping[c] = 1
                            else:
                                mapping[c] = 0
                        configs.append((model, fields, mapping))

    MODEL_BASE = "roberta"
    PROJECT_NAME = "SemEval23-classification"


    data_paths = {
        "train": "./data/train.jsonl",
        "validation":"./data/val.jsonl"}


    for model, fields, mapping in configs:
        file_name = config_name(model, fields, mapping)

        # ignore trained configs
        model_file_name = "-".join(file_name.split("-")[:-6])
        try:
            models_trained = os.listdir("./models")
            models_trained = [ "-".join(name.split("-")[:-6]) for name in models_trained]
        except:
            models_trained = []

        if model_file_name not in models_trained:

            print("Start training")
            print("Configs:", model, fields, mapping)
            print("Name:", file_name)

            # Load dataset
            dataset = DataLoader(fields=fields, files=data_paths, mapping=mapping)
            dataset = preprocess(dataset=dataset, model_base=MODEL_BASE, model_name=model)

            if model.endswith("full"):
                model = "./models" + model

            # Train
            trainer_trained = create_model(MODEL_BASE, model, file_name, dataset)
            print(trainer_trained.evaluate())


            trainer_trained.save_model(os.path.join('./models', file_name))

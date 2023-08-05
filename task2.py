### PyTerrier

import pyterrier as pt
import pandas as pd
from typing import List, Dict

def main():

    pt.init()

    def load_dataset(
            files: Dict[str, str] = {"train": "./data/ntest.jsonl",
                                     "validation": "./data/val.jsonl"}) -> pd.DataFrame:
        mapping: Dict[str, str] = {'passage': 0, 'phrase': 1, 'multi': 2}

        dataset = {}

        def load_data(file: str):
            df = pd.read_json(file, lines=True)
            return df

        for split in list(files.keys()):
            dataset[split] = load_data(files[split])

        return dataset

    dataset = load_dataset()

    for _, row in dataset["train"].iterrows():
        break
    print(dataset)

    def clean_string(s):
        return re.sub(r'[^\w\s]', '', s)

    passages = pd.DataFrame({
        "qid": "q1",
        "query": [clean_string(row["postText"][0]) for _ in range(len(row["targetParagraphs"]))],
        "docno": ["d" + str(i + 1) for i in range(len(row["targetParagraphs"]))],
        "text": [clean_string(paragraph) for paragraph in row["targetParagraphs"]],
    })

    textscorer = pt.text.scorer(body_attr="text", wmodel="DirichletLM")
    rtr = textscorer.transform(passages)
    rtr.sort_values("rank", ascending=True)

    res = []

    for _, row in dataset["train"].iterrows():
        import re

        def clean_string(s):
            return re.sub(r'[^\w\s]', '', s)

        passages = pd.DataFrame({
            "qid": "q1",
            "query": [clean_string(row["postText"][0]) for _ in range(len(row["targetParagraphs"]))],
            "docno": ["d" + str(i + 1) for i in range(len(row["targetParagraphs"]))],
            "text": [clean_string(paragraph) for paragraph in row["targetParagraphs"]],
        })

        textscorer = pt.text.scorer(body_attr="text", wmodel="BB2")
        rtr = textscorer.transform(passages)

        rtr.sort_values("rank", ascending=True)

        for index, _row in rtr.iterrows():
            if _row['rank'] == 0:
                res.append(_row['text'])

    print(res)

    import csv

    csv_file_name = "./data/out2.csv"
    i = 0
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "spoiler"])
        for string in res:
            writer.writerow([i, string])
            i = i + 1

if __name__ == '__main__':
    main()

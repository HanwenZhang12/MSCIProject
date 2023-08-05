### merge predicted tags to test file

import pandas as pd
import sys

if __name__ == '__main__':
    testFile = pd.read_json('./data/test.jsonl', lines=True)
    spoilerTypeFile = pd.read_json('./data/out.jsonl', lines=True)

    outputFile = pd.merge(testFile, spoilerTypeFile, on='id')

    outputFile.to_json('./data/ntest.jsonl', orient='records', lines=True)


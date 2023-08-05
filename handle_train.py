### Remove useless key-value in trian/val/test files

import jsonlines

if __name__ == '__main__':

    def filter_jsonl(input_path, output_path):
        with jsonlines.open(input_path, mode='r') as reader, jsonlines.open(output_path, mode='w') as writer:
            for obj in reader:
                if 'uuid' in obj:
                    del obj['uuid']
                if 'postId' in obj:
                    del obj['postId']
                if 'postPlatform' in obj:
                    del obj['postPlatform']
                if 'targetTitle' in obj:
                    del obj['targetTitle']
                if 'targetDescription' in obj:
                    del obj['targetDescription']
                if 'targetKeywords' in obj:
                    del obj['targetKeywords']
                if 'targetMedia' in obj:
                    del obj['targetMedia']
                if 'targetUrl' in obj:
                    del obj['targetUrl']
                if 'provenance' in obj:
                    del obj['provenance']
                if 'spoilerPositions' in obj:
                    del obj['spoilerPositions']
                if 'tags' in obj:
                    del obj['tags']
                writer.write(obj)

    input_path = './data/test.jsonl'
    output_path = './data/test2.jsonl'

    filter_jsonl(input_path, output_path)
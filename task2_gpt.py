### gpt3
# -*- coding: utf-8 -*-

import json

import os
import openai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha()]
    res = ''
    for token in tokens:
        res = res + token + ' '
    return res[:-1]

if __name__ == '__main__':

    openai.api_key = "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"

    def read_train_file():
        file_path = './data/test2.txt'
        train = []
        with open(file_path, "r",  encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                train.append(data)
        print('Train data has been read successfully.')
        return train

    train = read_train_file()

    res = []

    i=0
    failed=0

    for t in train:
        title = preprocess_sentence(t['postText'][0])
        body = ''
        for p in t['targetParagraphs']:
            body = body + p + '. '
        body = preprocess_sentence(body[:-1])

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[
                  {"role": "system", "content": "I will provide you with multiple sets of data. Each set of data consists of three elements: a title, a body, and a spoiler. Your task is to find the spoiler that corresponds to the given title. Please note that the spoiler must be a snippet taken from the body."},
                  {"role": "user", "content": "Now I will give you an example: title: Wes Welker Wanted Dinner With Tom Brady, But Patriots QB Had Better Idea. body: It’ll be just like old times this weekend for Tom Brady and Wes Welker. Welker revealed Friday morning on a Miami radio station that he contacted Brady because he’ll be in town for Sunday’s game between the New England Patriots and Miami Dolphins at Gillette Stadium. It seemed like a perfect opportunity for the two to catch up. But Brady’s definition of \"catching up\" involves far more than just a meal. In fact, it involves some literal \"catching\" as the Patriots quarterback looks to stay sharp during his four-game Deflategate suspension. I hit him up to do dinner Saturday night. He’s like, ‘I’m going to be flying in from Ann Arbor later (after the Michigan-Colorado football game), but how about that morning we go throw?’ \" Welker said on WQAM, per The Boston Globe. \"And I’m just sitting there, I’m like, ‘I was just thinking about dinner, but yeah, sure. I’ll get over there early and we can throw a little bit. Welker was one of Brady’s favorite targets for six seasons from 2007 to 2012. It’s understandable him and Brady want to meet with both being in the same area. But Brady typically is all business during football season. Welker probably should have known what he was getting into when reaching out to his buddy. That’s the only thing we really have planned,\" Welker said of his upcoming workout with Brady. \"It’s just funny. I’m sitting there trying to have dinner. ‘Hey, get your ass up here and let’s go throw.’ I’m like, ‘Aw jeez, man.’ He’s going to have me running like 2-minute drills in his backyard or something. Maybe Brady will put a good word in for Welker down in Foxboro if the former Patriots wide receiver impresses him enough. spoiler: how about that morning we go throw?"},
                  {"role": "user", "content": "title: "+ title + ".body: " + body}
                ]
            )

            res.append([i,completion.choices[0]["message"]["content"][8:]])
        except Exception as e:
            res.append([i,'failed'])
            failed+=1
        i=i+1
        if i%50 == 0:
          print("i="+str(i)+" failed="+str(failed))
    print("failed: " + str(failed))

    import csv

    filename = './data/out2.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in res:
            csv_writer.writerow(row)

import gensim
import json
from PreprocessingEnglish import Preprocessor
from collections import Counter
import itertools
import numpy as np
from random import randint

dataFoler = "dataSemEval"
trainingFile = dataFoler + "/Training_subtask2.txt"
testFile = dataFoler + "/Testing_subtask2.txt"

trainingSub1 = dataFoler + "/Training.txt"
testSub1 = dataFoler + "/Testing.txt"

ENTITY = []
ATTRIBUTE = []
with open("MetaData/entities", "r+") as file:
    line = file.readline()
    ENTITY = line.split(",")

with open("MetaData/attributes", "r+") as file:
    line = file.readline()
    ATTRIBUTE = line.split(",")

VOCAB = {}
VOCAB_INV = {}

with open("MetaData/vocab.txt", "r+") as datafile:
    data = json.load(datafile)
    VOCAB = data
with open("MetaData/vocab_inv.txt", "r+") as datafile:
    data = json.load(datafile)
    VOCAB_INV = data

dummy_word = "null_word"
tmp_sequence_length = 500


def load_all_data_single_aspect_train(aspect_name, type):

    x,y = load_all_data_single_aspect(aspect_name, trainingFile, type)
    if aspect_name == 'LAPTOP':
        return x,y

    x_e, y_e = load_extra_from_sub1(aspect_name, type)
    x.extend(x_e)
    y.extend(y_e)
    return x,y

def load_all_data_single_aspect_test(aspect_name, type):
    return load_all_data_single_aspect(aspect_name, testFile, type)

def load_all_data_single_aspect(aspect_name, file, type):
    x = []
    y = []

    if type == 'entity' :
        if aspect_name not in ENTITY:
            return x, y

    with open(file) as data_file:
        data = json.load(data_file)

    for review in data["Reviews"]["Review"]:

        #Read Text
        text_data = []
        if isinstance(review["sentences"]["sentence"], dict):
            sentences = [review["sentences"]["sentence"]]
        else:
            sentences = review["sentences"]["sentence"]

        for sentence in sentences:
            text_data.extend(Preprocessor.tokenizeOneSentence(sentence['text']))

        #Read Label
        if "Opinions" not in review:
            continue

        if isinstance(review["Opinions"]["Opinion"], dict):
            opinions = [review["Opinions"]["Opinion"]]
        else:
            opinions = review["Opinions"]["Opinion"]
        isMentioned = False

        for opinion in opinions:
            entity = opinion["-category"].split("#")[0]
            attribute = opinion["-category"].split("#")[1]
            polarity = opinion["-polarity"]

            if type == 'entity' :
                if entity == aspect_name:
                    isMentioned = True
            if type == 'attribute' :
                if attribute == aspect_name:
                    isMentioned = True

        x.append(text_data)

        if isMentioned:
            y.append(1)
        else:
            y.append(0)


    # print ("Data variation for label " + aspect_name)
    # print ("Possitive : " + str(y.count([0,1])))
    # print ("Negative : " + str(y.count([1,0])))
    return x, y



def build_input_data(sentences, labels):
    x = np.array([[get_index_in_vocab(word) for word in sentence] for sentence in sentences])
    x = np.asarray(x)
    y = np.array(labels)
    return x, y

def pad_sentences(sentences, padding_word="<PAD/>"):

    sequence_length = tmp_sequence_length
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def get_index_in_vocab(word):
    if word in VOCAB:
        return VOCAB[word]
    else:
        return VOCAB[dummy_word]

def load_data(aspect, type):
    x, y = load_all_data_single_aspect_train(aspect, type)
    x = pad_sentences(x)
    x, y = build_input_data(x, y)

    x_t, y_t = load_all_data_single_aspect_test(aspect, type)
    x_t = pad_sentences(x_t)
    x_t, y_t = build_input_data(x_t, y_t)

    return x, y, x_t, y_t


def load_extra_from_sub1 (aspect_name, type):
    x_pos = []
    x_neg = []

    if type == 'entity':
        if aspect_name not in ENTITY:
            return x, y
    if type == 'attribute':
        if aspect_name not in ATTRIBUTE:
            return x, y

    with open(trainingSub1) as data_file:
        data = json.load(data_file)

    for review in data["Reviews"]["Review"]:

        if isinstance(review["sentences"]["sentence"], dict):
            sentences = [review["sentences"]["sentence"]]
        else:
            sentences = review["sentences"]["sentence"]
        for sentence in sentences:
            opinions = []
            if "Opinions" not in sentence:
                continue
            if isinstance(sentence["Opinions"]["Opinion"], dict):
                opinions = [sentence["Opinions"]["Opinion"]]
            else:
                opinions = sentence["Opinions"]["Opinion"]

            isMentioned = False
            for opinion in opinions:
                entity = opinion["-category"].split("#")[0]
                attribute = opinion["-category"].split("#")[1]

                if type == 'entity':
                    if entity == aspect_name:
                        isMentioned = True
                if type == 'attribute':
                    if attribute == aspect_name:
                        isMentioned = True
            if isMentioned:
                x_pos.append(Preprocessor.tokenizeOneSentence(sentence['text']))
            else:
                x_neg.append(Preprocessor.tokenizeOneSentence(sentence['text']))

    output_sentences = []
    output_label = []

    if len(x_pos) == 0:
        return [], []

    for i in range(100):
        pos_sentence = x_pos[randint(0, len(x_pos)-1)]
        pos_sentence2 = x_pos[randint(0, len(x_pos)-1)]
        neg_sentence1 = x_neg[randint(0, len(x_neg)-1)]
        neg_sentence2 = x_neg[randint(0, len(x_neg)-1)]
        neg_sentence3 = x_neg[randint(0, len(x_neg) - 1)]

        output_sentences.append(pos_sentence + neg_sentence1 + neg_sentence2)
        output_label.append(1)
        output_sentences.append(neg_sentence1 + pos_sentence + neg_sentence2)
        output_label.append(1)
        output_sentences.append(pos_sentence  + pos_sentence2 + neg_sentence2)
        output_label.append(1)
        output_sentences.append(neg_sentence1 + neg_sentence2 + pos_sentence2)
        output_label.append(1)

        output_sentences.append(neg_sentence1 + neg_sentence2 + neg_sentence3)
        output_label.append(0)

    return output_sentences, output_label



def load_sentence_only_test () :
    x, y, x_t, y_t = load_data(ENTITY[1], 'entity')
    return x_t




import random
if __name__ == "__main__":
    x, y, x_t, y_t = load_data(ENTITY[1], 'entity')

    c = list(zip(x, y))

    random.shuffle(c)

    x, y = zip(*c)
    print np.asarray(x)








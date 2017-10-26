
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
BOTH = []
with open("MetaData/entities", "r+") as file:
    line = file.readline()
    ENTITY = line.split(",")

with open("MetaData/attributes", "r+") as file:
    line = file.readline()
    ATTRIBUTE = line.split(",")

with open("MetaData/entities_attributes_subtask2(2)", "r+") as file:
    line = file.readline()
    BOTH = line.split(",")

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


def load_all_data_single_aspect_train(aspect_name):

    x,y = load_all_data_single_aspect(aspect_name, trainingFile)
    if aspect_name == 'LAPTOP':
        return x,y

    x_e, y_e = load_extra_from_sub1(aspect_name)
    x.extend(x_e)
    y.extend(y_e)
    return x,y

def load_all_data_single_aspect_test(aspect_name):
    return load_all_data_single_aspect(aspect_name, testFile)

def load_all_data_single_aspect(aspect_name, file):
    x = []
    y = []



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


        hasEntity = False
        hasAttribute = False

        for opinion in opinions:
            both = opinion["-category"]
            entity = opinion["-category"].split("#")[0]
            attribute = opinion["-category"].split("#")[1]
            polarity = opinion["-polarity"]

            #aspect co dang both : e#a
            if entity == aspect_name.split('#')[0]:
                hasEntity = True
                if attribute == aspect_name.split('#')[1]:
                    hasAttribute = True

        if hasEntity:
            x.append(text_data)
            if hasAttribute:
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

def load_data(aspect):
    x, y = load_all_data_single_aspect_train(aspect)
    x = pad_sentences(x)
    x, y = build_input_data(x, y)

    x_t, y_t = load_all_data_single_aspect_test(aspect)
    x_t = pad_sentences(x_t)
    x_t, y_t = build_input_data(x_t, y_t)

    return x, y, x_t, y_t


def load_extra_from_sub1 (aspect_name):
    mention_true = []
    mention_false = []
    not_mention = []



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

            hasEntity = False
            hasAttribute = False

            for opinion in opinions:
                both = opinion["-category"]
                entity = opinion["-category"].split("#")[0]
                attribute = opinion["-category"].split("#")[1]
                polarity = opinion["-polarity"]

                # aspect co dang both : e#a
                if entity == aspect_name.split('#')[0]:
                    hasEntity = True
                    if attribute == aspect_name.split('#')[1]:
                        hasAttribute = True

            if hasEntity:
                if hasAttribute:
                    mention_true.append(Preprocessor.tokenizeOneSentence(sentence['text']))
                else:
                    mention_false.append(Preprocessor.tokenizeOneSentence(sentence['text']))

            else :
                not_mention.append(Preprocessor.tokenizeOneSentence(sentence['text']))

    maximum_loop = len(mention_true)*4

    output_sentences = []
    output_label = []

    if len(mention_true) == 0:
        return [], []
    if len(mention_false) == 0:
        return [], []

    for i in range(min(80,maximum_loop)):
        true_1 = mention_true[randint(0, len(mention_true)-1)]
        true_2 = mention_true[randint(0, len(mention_true)-1)]
        false_1 = mention_false[randint(0, len(mention_false)-1)]
        false_2 = mention_false[randint(0, len(mention_false)-1)]
        not_1 = not_mention[randint(0, len(not_mention) - 1)]
        not_2 = not_mention[randint(0, len(not_mention) - 1)]
        not_3 = not_mention[randint(0, len(not_mention) - 1)]

        output_sentences.append(true_1 + true_2 + false_1)
        output_label.append(1)
        output_sentences.append(true_1 + false_1 + not_1)
        output_label.append(1)
        output_sentences.append(true_2  + not_3 + not_2)
        output_label.append(1)
        output_sentences.append(false_1 + false_2 + not_1)
        output_label.append(0)
        output_sentences.append(not_1 + false_1 + not_3)
        output_label.append(0)
        output_sentences.append(not_3 + not_1 + false_2)
        output_label.append(0)

    return output_sentences, output_label






import random
if __name__ == "__main__":
    x, y, x_t, y_t = load_data(BOTH[3])

    c = list(zip(x, y))

    random.shuffle(c)

    x, y = zip(*c)
    print np.asarray(y).tolist().count(1)
    print np.asarray(y).tolist().count(0)








# -*- coding: ascii -*-
import os, sys

import pandas as pd

import csv

import string

import nltk

from time import time
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

import json

import operator


stopWords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "alone", "along", "already","am","among", "amongst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "by", "call", "co", "con", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "few", "fill", "find", "fire", "first", "for", "former", "formerly", "found", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

class Preprocessor:
    lemmatizer = WordNetLemmatizer()
    stop_word = stopWords

    @staticmethod
    def tokenizeMutipleSentences(doc):

        listToken = []

        try:
            sent_tokenize(doc)
        except Exception as e:
            print ("cant tokenize : " + doc + str(e))
            return []

        for sent in sent_tokenize(doc):
            listToken.extend(Preprocessor.tokenizeOneSentence(sent))
            listToken.append(".")

        return listToken


    @staticmethod
    def tokenizeOneSentence(sentence):

        listToken = []
        token_iter = iter(pos_tag(wordpunct_tokenize(sentence)))

        for token, tag in token_iter:

            # ignore hidden name
            if token[0] == '@':
                next(token_iter)

            # clean up some token
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            try:
                token = token.encode('utf-8')
            except Exception:
                continue

            # If stopword, ignore token and continue
            if token in set(Preprocessor.stop_word):
                continue

            # If punctuation, ignore token and continue
            if all(char in set(string.punctuation) for char in token):
                continue

            # Lemmatize token
            try:
                lemmatized_token = Preprocessor.__lemmatize(token, tag)
            except Exception as e:
                continue

            listToken.append(lemmatized_token);

        return listToken

    @staticmethod
    def __lemmatize(token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return Preprocessor.lemmatizer.lemmatize(token, tag)




# AutoGen file data
# Xoa data trong tokenized_data truoc khi train lai, neu khong la no de len


def read_subtask1 () :
    dataFolder = "dataSemEval"
    trainingFile = dataFolder + "/Training.txt"
    list_label = {}

    with open(trainingFile) as data_file:
        data = json.load(data_file)

    entities = []
    attributes = []
    boths = []
    everything_map = {}
    no_sentiment_map = {}
    total = 0

    for review in data["Reviews"]["Review"]:

        if isinstance(review["sentences"]["sentence"], dict):
            sentences = [review["sentences"]["sentence"]]
        else:
            sentences = review["sentences"]["sentence"]

        for sentence in sentences:
            if "Opinions" not in sentence:
                continue

            if isinstance(sentence["Opinions"]["Opinion"], dict):
                opinions = [sentence["Opinions"]["Opinion"]]
            else:
                opinions = sentence["Opinions"]["Opinion"]

            isMentioned = False

            total += 1
            for opinion in opinions:

                both = opinion['-category']
                entity = opinion["-category"].split("#")[0]
                attribute = opinion["-category"].split("#")[1]
                polarity = opinion['-polarity']

                chosen = both

                if (chosen ) not in no_sentiment_map:
                    no_sentiment_map[chosen ] = 1
                else:
                    no_sentiment_map[chosen ] = no_sentiment_map[chosen ] + 1

                if polarity == 'positive':
                    if (chosen + '-pos') not in everything_map:
                        everything_map[chosen + '-pos'] = 1
                    else:
                        everything_map[chosen + '-pos'] = everything_map[chosen + '-pos'] + 1
                else:
                    if (chosen + '-neg') not in everything_map:
                        everything_map[chosen + '-neg'] = 1
                    else:
                        everything_map[chosen + '-neg'] = everything_map[chosen + '-neg'] + 1

    sorted_everything_map = sorted(everything_map.items(), key=operator.itemgetter(0))
    print sorted_everything_map

    print no_sentiment_map
    print total

                        # Read Label

def returnNotMatches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]

if __name__ == "__main__":
    dataFolder = "dataSemEval"
    trainingFile = dataFolder + "/Training_subtask2.txt"
    list_label = {}

    with open(trainingFile) as data_file :
        data = json.load(data_file)


    entities = []
    attributes = []
    boths = []

    everything_map = {}
    count_map = {}
    for review in data["Reviews"]["Review"] :



        opinions = []

        if isinstance(review["Opinions"]["Opinion"], dict):
            opinions = [review["Opinions"]["Opinion"]]

        else :
            opinions = review["Opinions"]["Opinion"]



        for opinion in opinions :

                both = opinion['-category']

                entity = opinion["-category"].split("#")[0]
                attribute = opinion["-category"].split("#")[1]

                polarity = opinion['-polarity']
                chosen = both

                if chosen in count_map:
                    count_map[chosen] = count_map[chosen] + 1
                else :
                    count_map[chosen] = 1



                if polarity == 'positive' :
                    if (chosen + '-pos') not in everything_map :
                        everything_map[chosen + '-pos'] = 1
                    else :
                        everything_map[chosen + '-pos'] = everything_map[chosen + '-pos'] +1
                else :
                    if (chosen + '-neg') not in everything_map :
                        everything_map[chosen + '-neg'] = 1
                    else :
                        everything_map[chosen + '-neg'] = everything_map[chosen + '-neg'] +1


                if opinion["-category"] not in boths:
                    boths.append(opinion["-category"])
                if entity not in entities :
                    entities.append(entity)
                if attribute not in attributes:
                    attributes.append(attribute)

                if attribute not in list_label :
                    list_label[attribute] = 1
                else :
                    list_label[attribute] = list_label[attribute] + 1

    # sorted_map = sorted(count_map.items(), key=operator.itemgetter(0))
    #
    # sorted_x = sorted(list_label.items(), key=operator.itemgetter(1))
    # print sorted_x
    #
    # sorted_everything_map = sorted(everything_map.items(), key=operator.itemgetter(0))
    # print sorted_everything_map


    # with open ('MetaData/entities_subtask2(2)', 'w+') as file :
    #      file.write(",".join(entities))
    # with open ('MetaData/attributes_subtask2(2)', 'w+') as file :
    #      file.write(",".join(attributes))
    # with open ('MetaData/entities_attributes_subtask2(2)', 'w+') as file :
    #      file.write(",".join(boths))


    with open ('MetaData/entities_attributes_subtask2', 'r') as data_wrong :
        lines = data_wrong.readline()
        wrong = lines.split(",")

    with open ('MetaData/entities_attributes_subtask2(2)', 'r') as data_true :
        lines = data_true.readline()
        true = lines.split(",")

    thieu, thua = returnNotMatches(true, wrong)

    print thieu
    print thua
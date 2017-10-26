import json

from keras.models import load_model
import numpy as np
import os


testFile = 'dataSemEval/' + "/Testing_subtask2.txt"
jsonFile = 'Result/JSON'

with open(testFile) as data_file :
    data_test = json.load(data_file)


sentiment_model_map = {}

from dataHelper_entity import load_sentence_only_test
test_sen = load_sentence_only_test()

for i in range(0,79):
    print ('Calculating attribute for sentence no : ' + str(i))

    review = data_test["Reviews"]["Review"][i]

    if isinstance(review["Opinions"]["Opinion"], dict):
        opinions = [review["Opinions"]["Opinion"]]
    else:
        opinions = review["Opinions"]["Opinion"]

    for opinion in opinions:
        category = opinion['-category']
        entity = category.split('#')[0]
        attribute = category.split('#')[1]

        opinion['-polarity'] = 'negative'

        sentimentFolder = 'TrainedModel/sentiment/' + entity

        sentiment_model = None


        if not os.path.exists(sentimentFolder + '/' + str(attribute) + '.h5'):
            continue

        if category in sentiment_model_map:
            sentiment_model = sentiment_model_map[category]
        else:
            sentiment_model = load_model(sentimentFolder + '/' + str(attribute) +'.h5')
            sentiment_model_map[category] = sentiment_model

        polarity = sentiment_model.predict(np.asarray([test_sen[i]]))
        sentiment_output = 'negative'

        if polarity[0] > 0.6:
            sentiment_output = 'positive'

        opinion['-polarity'] = sentiment_output



print str(data_test).encode("ascii")

with open('Result/final_JSON_sentiment', 'w+') as outfile:
    json.dump(data_test, outfile)

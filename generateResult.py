
import json
from dataHelper_entity import load_sentence_only_test, load_data
from keras.models import Model
from sklearn.metrics import recall_score, precision_score, accuracy_score
from keras.models import load_model
import numpy as np
import os
from os import path

defaultAttribute = {
    'BATTERY' : 'OPERATION_PERFORMANCE',
    'COMPANY' : 'GENERAL',
    'CPU' : 'OPERATION_PERFORMANCE',
    'DISPLAY' : 'QUALITY',
    'GRAPHICS' : 'GENERAL',
    'HARD_DISC' : 'QUALITY',
    'KEYBOARD' : 'DESIGN_FEATURES',
    'LAPTOP' : 'GENERAL',
    'MEMORY' : 'DESIGN_FEATURES',
    'MOTHERBOARD' : 'QUALITY',
    'MOUSE' : 'USABILITY',
    'MULTIMEDIA_DEVICES' : 'GENERAL',
    'OS': 'GENERAL',
    'POWER_SUPPLY' : 'QUALITY',
    'SHIPPING' : 'QUALITY',
    'SOFTWARE' : 'GENERAL',
    'SUPPORT' : 'QUALITY',
    'WARRANTY' : 'GENERAL'
}

dataFoler = "dataSemEval"
trainingFile = dataFoler + "/Training_subtask2.txt"
testFile = dataFoler + "/Testing_subtask2.txt"

trainingSub1 = dataFoler + "/Training.txt"
testSub1 = dataFoler + "/Testing.txt"

ENTITY = ['COMPANY','LAPTOP', 'BATTERY', 'HARD_DISC', 'DISPLAY', 'MULTIMEDIA_DEVICES', 'OS', 'MOUSE', 'KEYBOARD', 'SUPPORT', 'MEMORY', 'CPU', 'SOFTWARE', 'GRAPHICS', 'MOTHERBOARD']
BOTH = []
ATTRIBUTE = []
with open("MetaData/attributes", "r+") as file:
    line = file.readline()
    ATTRIBUTE = line.split(",")

with open("MetaData/entities_attributes_subtask2", "r+") as file:
    line = file.readline()
    BOTH = line.split(",")


with open(testFile) as data_file :
    data_test = json.load(data_file)

x, y, x_test, y_test = load_data('COMPANY', 'entity')

result = []


test_sen = load_sentence_only_test()

list_result = [dict() for x in range(0,79)]

for entity in ENTITY :
    print "labeling for entity : " + entity
    model = load_model('TrainedModel/entity/' + entity + '.h5')

    output_for_entity = model.predict(test_sen)

    test_round = []

    for num in output_for_entity:
        if num < 0.5:
            test_round.append(0)
        else:
            test_round.append(1)


    for i in range (0,79):
        if test_round[i] == 1:
            list_result[i][entity] = {}

both_model_map = {}
sentiment_model_map = {}

count = 0
for sentence_entity_map in list_result :
    print ('Calculating attribute for sentence no : ' + str(count))

    for entity, attribute_map in sentence_entity_map.iteritems():
        if attribute_map == None :
            continue

        attributeFolder = 'TrainedModel/attribute/' + entity
        list_possible_saved_model = [f for f in os.listdir(attributeFolder)]

        sentimentFolder = 'TrainedModel/sentiment/' + entity

        for model_attribute_name in list_possible_saved_model:

            attribute = model_attribute_name.split('.')[0]
            both = entity + "#" + attribute

            model = None
            if both in both_model_map :
                model = both_model_map[both]
            else :
                model = load_model(attributeFolder + '/' + str(model_attribute_name))
                both_model_map[both] = model


            y = model.predict(np.asarray([test_sen[count]]))
            if y[0] > 0.5:
                #load sentiment
                sentiment_model = None
                if both in sentiment_model_map :
                    sentiment_model = sentiment_model_map[both]
                else :
                    sentiment_model = load_model(sentimentFolder + '/' + str(model_attribute_name))
                    sentiment_model_map[both] = sentiment_model

                polarity = sentiment_model.predict(np.asarray([test_sen[count]]))
                sentiment_output = 'negative'

                if polarity[0] > 0.5:
                    sentiment_output = 'positive'
                
                    

                attribute_map [attribute] = sentiment_output

    if attribute_map != None :
        if len(attribute_map) == 0:
            defaultAttribute_current = defaultAttribute[entity]
            both = entity + '#' + defaultAttribute_current

            sentiment_model = None
            if both in sentiment_model_map:
                sentiment_model = sentiment_model_map[both]
            else:
                sentiment_model = load_model(sentimentFolder + '/' + str(model_attribute_name))
                sentiment_model_map[both] = sentiment_model

            polarity = sentiment_model.predict(np.asarray([test_sen[count]]))
            sentiment_output = 'negative'

            if polarity[0] > 0.5:
                sentiment_output = 'positive'

            attribute_map[defaultAttribute_current] = sentiment_output


    count += 1

print list_result

with open('Result/JSON', 'w+') as outfile:
    json.dump(list_result, outfile)










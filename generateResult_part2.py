import json


testFile = 'dataSemEval/' + "/Test_Only.txt"
jsonFile = 'Result/JSON'

with open(testFile) as data_file :
    data_test = json.load(data_file)

with open(jsonFile) as data_file:
    label = json.load(data_file)


for i in range(0,79):

    label_to_add = []

    label_row = label[i]
    for entity, attribute_map in label_row.iteritems():
        for attribute, sentiment in attribute_map.iteritems():
            label_to_add.append(entity + '#' + attribute + "ww" + sentiment)

    review = data_test["Reviews"]["Review"][i]
    review["Opinions"] = dict()
    review["Opinions"]["Opinion"] = []

    opinion_array = review["Opinions"]["Opinion"]

    for label_item in label_to_add:
        map = {}
        map['-category'] = label_item.split('ww')[0]
        map['-polarity'] = label_item.split('ww')[1]
        opinion_array.append(map)

print str(data_test).encode("ascii")

with open('Result/final_JSON', 'w+') as outfile:
    json.dump(data_test, outfile)



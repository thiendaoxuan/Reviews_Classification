import json
from PreprocessingEnglish import Preprocessor
from collections import Counter
import itertools


dataFoler = "dataSemEval"
trainingSub2 = dataFoler + "/Training_subtask2.txt"
testSub2 = dataFoler + "/Testing_subtask2.txt"

trainingSub1 = dataFoler + "/Training.txt"
testSub1 = dataFoler + "/Testing.txt"

dummy_word = "null_word"

def load_all_sentence (file):
    output = []
    with open(file) as data_file:
        data = json.load(data_file)

    for review in data["Reviews"]["Review"]:

        if isinstance(review["sentences"]["sentence"], dict):
            sentences = [review["sentences"]["sentence"]]
        else:
            sentences = review["sentences"]["sentence"]

        for sentence in sentences:
            output.append(Preprocessor.tokenizeOneSentence(sentence['text']))
    return output

def load_sentence_2_file():
    output_1 = load_all_sentence(trainingSub1)
    output_2 = load_all_sentence(trainingSub2)

    output_1.extend(output_2)
    return output_1

def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    list_vocabulary_inv = [output[0] for output in word_counts.most_common()]
    list_vocabulary_inv.append(dummy_word)
    list_vocabulary_inv = list(sorted(list_vocabulary_inv))

    vocabulary = {x: i for i, x in enumerate(list_vocabulary_inv)}
    vocabulary_inv = {i : x for i, x in enumerate(list_vocabulary_inv)}

    return vocabulary, vocabulary_inv

def create_dictionary_files ():
    x = load_sentence_2_file()
    vocab, vocab_inv = build_vocab(x)

    json_str_vocab =    json.dumps(vocab)
    json_str_vocab_inv = json.dumps(vocab_inv)

    json_vocab = json.loads(json_str_vocab)
    json_vocab_inv = json.loads(json_str_vocab_inv)

    with open('MetaData/vocab.txt', 'w+') as outfile:
        json.dump(json_vocab, outfile)
    with open('MetaData/vocab_inv.txt', 'w+') as outfile:
        json.dump(json_vocab_inv, outfile)

if __name__ == "__main__":
    create_dictionary_files()
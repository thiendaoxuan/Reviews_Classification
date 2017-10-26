# This Python file uses the following encoding: utf-8

from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from dataHelper_entity import load_data,ENTITY, tmp_sequence_length, VOCAB, VOCAB_INV
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import recall_score, precision_score, accuracy_score
import random

# x, y, x_test, y_test, vocabulary, vocabulary_inv = load_data(ENTITY[0], type = 'entity')

# sequence_length = tmp_sequence_length
# vocabulary_size = len(vocabulary_inv)
# embedding_dim = 256
# filter_sizes = [3, 4, 5]
# num_filters = 128
# drop = 0.5
# number_of_labels = 1
# nb_epoch = 4
# batch_size = 50
#
# # this returns a tensor
# inputs = Input(shape=(sequence_length,), dtype='int32')
# embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
# reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)
#
# conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal',
#                        activation='relu', dim_ordering='tf')(reshape)
# conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal',
#                        activation='relu', dim_ordering='tf')(reshape)
# conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal',
#                        activation='relu', dim_ordering='tf')(reshape)
#
# maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), border_mode='valid',
#                          dim_ordering='tf')(conv_0)
# maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), border_mode='valid',
#                          dim_ordering='tf')(conv_1)
# maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), border_mode='valid',
#                          dim_ordering='tf')(conv_2)
#
# merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
# flatten = Flatten()(merged_tensor)u'OPTICAL_DRIVES
# # reshape = Reshape((3*num_filters,))(merged_tensor)
# dropout = Dropout(0.5)(flatten)
#
# ############################
# # hidden = Dense(200, activation='relu')(drop1)
# # dropout = Dropout(0.5)(hidden)
# #############################
#
# # output = Dense(output_dim=13, activation='softmax')(dropout)
# output = Dense(output_dim=number_of_labels, activation='sigmoid')(dropout)
#
# # this creates a model that includes
# model = Model(input=inputs, output=output)
#
#
# # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




# starts training

avg_acc = 0
avg_pre = 0
avg_rec = 0
total_test = 0

for aspect in ENTITY:

    # if aspect == 'LAPTOP' :
    #     continue

    x, y, x_test, y_test = load_data(aspect, 'entity')
    c = list(zip(x,y))

    random.shuffle(c)

    x,y = zip(*c)
    x= np.asarray(x)
    y= np.asarray(y)



    vocabulary = VOCAB

    sequence_length = tmp_sequence_length
    vocabulary_size = len(vocabulary)
    embedding_dim = 256
    filter_sizes = [3, 4, 5]
    num_filters = 128
    drop = 0.5
    number_of_labels = 1
    nb_epoch = 4
    batch_size = 50

    # this returns a tensor
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

    conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal',
                           activation='relu', dim_ordering='tf')(reshape)
    conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal',
                           activation='relu', dim_ordering='tf')(reshape)
    conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal',
                           activation='relu', dim_ordering='tf')(reshape)

    maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), border_mode='valid',
                             dim_ordering='tf')(conv_2)

    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
    flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3*num_filters,))(merged_tensor)
    dropout = Dropout(0.5)(flatten)

    ############################
    # hidden = Dense(200, activation='relu')(drop1)
    # dropout = Dropout(0.5)(hidden)
    #############################

    # output = Dense(output_dim=13, activation='softmax')(dropout)
    output = Dense(output_dim=number_of_labels, activation='sigmoid')(dropout)

    # this creates a model that includes
    model = Model(input=inputs, output=output)

    # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, callbacks=[],
          validation_data=(x_test, y_test))

    model.save('TrainedModel/entity/' + aspect + '.h5')
    test_result = model.predict(x_test)

    test_result = model.predict(x_test)

    test_round = []

    for data in test_result:
        if data < 0.5:
            test_round.append(0)
        else:
            test_round.append(1)

    test_round = np.asarray(test_round)

    test_result = test_round


    print ("-------------------------")

    print("Precision "  + aspect)
    print (precision_score(y_test,test_result))
    print("Recall "  + aspect)
    print (recall_score(y_test,test_result))









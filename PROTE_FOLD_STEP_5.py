# Zachary Webel
# zow2@georgetown.edu

'''
Step 5 creates the model, trains and evaulates
'''
#########################################################################################


# Imports
from tensorflow.keras.layers import Dense, Flatten, LSTM, Concatenate, BatchNormalization,Input, Dropout,Conv2D,Attention, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow as tf
from keras import optimizers
import numpy as np
import pandas as pd


# define the shapes of the inputs
seq_input = Input(shape=(seq_length, 3))
prop_input = Input(shape=(9,))


# define strategy to train on multiple gpus
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
with strategy.scope():

    # BRANCH 1: Sequential language 
    lstm = LSTM(3, return_sequences=True)(seq_input)
    lstm = LSTM(6, return_sequences=True)(lstm)
    #only return last hidden state
    lstm = LSTM(12, return_sequences=False)(lstm)

    # BRANCH 2: Self Attention
    attention = Attention()([seq_input, seq_input])
    attention_flatten = Flatten()(attention)

    # BRANCH 3: Expanded Property transformation
    properties = Dense(16,activation='relu')(prop_input)
    properties = Dense(16,activation='relu')(properties)
    properties = Dense(16,activation='relu')(properties)
    
    # Concat the outputs of each branch
    concatenation = Concatenate()([lstm, attention_flatten,properties])

    # Normalize the concated vector
    normalized = BatchNormalization()(concatenation)

    # Stacked dense-dropout layers
    dense = Dense(1024, activation='relu')(normalized)
    dropout = Dropout(0.2)(dense)
    dense = Dense(1024, activation='relu')(dropout)
    dropout = Dropout(0.2)(dense)
    dense = Dense(1024, activation='relu')(dropout)
    dropout = Dropout(0.2)(dense)
    dense = Dense(1024, activation='relu')(dropout)
    dropout = Dropout(0.2)(dense)
    dense = Dense(1024, activation='relu')(dropout)
    dropout = Dropout(0.2)(dense)
    dense = Dense(1024, activation='relu')(dropout)
    dropout = Dropout(0.2)(dense)

    # Output 20 amino acids, 1 stop tag
    # Probability distribution
    output = Dense(21,activation='softmax')(dense)

    # Create model
    model = Model(inputs=[seq_input,prop_input], outputs=output)
    opt = optimizers.Adam(learning_rate=0.0001)

    # compile model
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics = ['accuracy'])

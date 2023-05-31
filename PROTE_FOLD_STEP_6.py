# Zachary Webel
# zow2@georgetown.edu

'''
Step 6 generates novel sequences with desired secondary structure fractions
'''
#########################################################################################


# Imports
import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from random import randint
from tensorflow.keras.layers import Dense, Flatten, LSTM, Concatenate, BatchNormalization,Input, Dropout,Conv2D,Attention, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow as tf
from keras import optimizers


#import embeddings
embedding_matrix = pd.read_csv('/aa_embeddings.csv')
embedding_matrix.columns=['amino_acid','x1','x2','x3']


# define amino acid list, action space, seq len and step size
amino_acid_list = list(embedding_matrix['amino_acid'])
action_space =  list(embedding_matrix['amino_acid'])
action_space.append('stop')
seq_length = 10
step_size = 1



# Load model
seq_input = Input(shape=(seq_length, 3))
prop_input = Input(shape=(9,))

lstm = LSTM(3, return_sequences=True)(seq_input)
lstm = LSTM(6, return_sequences=True)(lstm)
lstm = LSTM(12, return_sequences=False)(lstm)

attention = Attention()([seq_input, seq_input])
attention_flatten = Flatten()(attention)

properties = Dense(16,activation='relu')(prop_input)
properties = Dense(16,activation='relu')(properties)
properties = Dense(16,activation='relu')(properties)

concatenation = Concatenate()([lstm, attention_flatten,properties])
normalized = BatchNormalization()(concatenation)

dense = Dense(1024, activation='relu',trainable=False)(normalized)
dropout = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu',trainable=False)(dropout)
dropout = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu',trainable=False)(dropout)
dropout = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu',trainable=False)(dropout)
dropout = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu',trainable=False)(dropout)
dropout = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu',trainable=False)(dropout)
dropout = Dropout(0.2)(dense)
output = Dense(21,activation='softmax',trainable=False)(dense)


model = Model(inputs=[seq_input,prop_input], outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.load_weights("prote_fold_model.h5")




# Model trained on data from phage tail fiber seqs. Let's see if it works on antibody seqs
# from another of my projects
testing_seqs = pd.read_csv('antibody_database.csv')


# I will use desired structure percentages from these antibody sequences
testing_properties = []
for sequence in testing_seqs['sequence']:
    if(len(sequence))>=60:
        analysed_seq = ProteinAnalysis(Seq(sequence))
        secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
        testing_properties.append([secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]])




# Generate 200 sequences
generated_sequences = []
for count in range(200):
    
    # Random target features
    target_features = testing_properties[randint(0,len(testing_properties)-1)]
    # or define your own helix,turn, sheet
    #target_features = [0.25, 0.18, 0.4]
    
    # RANDOM SEED
    generated_sequence = []
    for i in range(seq_length):
        generated_sequence.append(amino_acid_list[randint(0,len(amino_acid_list)-1)])
        

    # initialize current positon and stop flag
    current_position = 0
    stop_flag = False
    while stop_flag == False and len(generated_sequence)<=300:

        # window sequence
        input_sequence = generated_sequence[current_position: current_position + seq_length]
        model_input = []

        # sequence inpu
        model_input.append([embedding_matrix.loc[embedding_matrix['amino_acid'] == aa].values[0][1:] if aa in amino_acid_list else embedding_matrix.loc[embedding_matrix['amino_acid'] == amino_acid_list[randint(0,len(amino_acid_list)-1)]].values[0][1:] for aa in input_sequence])
        X = np.reshape(model_input[:1], (1, seq_length,3))
        X = np.array(X).astype(float)


        # WINDOW FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
        ###################################################################################
        analysed_seq = ProteinAnalysis(Seq("".join(input_sequence)))
        secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
        window_helix, window_turn, window_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
        window_features = [window_helix,window_turn,window_sheet]
        ###################################################################################


        # CURRENT_BUILT FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
        ###################################################################################
        analysed_seq = ProteinAnalysis(Seq("".join(generated_sequence)))
        secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
        current_built_helix, current_built_turn, current_built_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
        current_built_features = [current_built_helix,current_built_turn,current_built_sheet]
        target_current_difference = [target_features[i]-current_built_features[i] for i in range(len(target_features))]
        ###################################################################################


        #property feature vector
        property_vector = [target_features+window_features+target_current_difference]
        property_vector = np.array(property_vector)

        # predict the distribution of the next amio acid or stop 
        predicted_distribution = model.predict([X,property_vector],verbose=0)

        # use numpy random choice to sample
        sampled_action = np.random.choice(action_space,p=predicted_distribution[0])

        # if we sample a stop flag, then stop
        if(sampled_action=='stop'):
            stop_flag = True
        else:
            generated_sequence = generated_sequence + [sampled_action]
        current_position = current_position + 1


    # create string representation of the generated protein given a grace period of 10 amino acids
    # 30 random seed + 10 aa grace period so we start at the 40th aa
    protein = ""
    for amino_acid in generated_sequence:
        protein = protein + amino_acid
    # generated FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
    ###################################################################################
    protein = protein[40:]
    # check for reasonable length of 60 aa
    if(len(protein)>=60):

        # calculate properties
        analysed_seq = ProteinAnalysis(Seq(protein))
        secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
        generated_helix, generated_turn, generated_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
        generated_features = [generated_helix,generated_turn,generated_sheet]
        target_generated_difference = [(abs(target_features[feature]-generated_features[feature])) / (abs(target_features[feature]+generated_features[feature]) / 2) * 100 for feature in range(len(target_features))]
        ###################################################################################
        print(protein,'\n')
        #print('Avg abs % diff: ', sum(target_generated_difference)/len(target_generated_difference))
        #for difference in target_generated_difference:
            #print(difference)
        print('Target: ',target_features)
        print('helix: ',generated_helix)
        print('turn: ',generated_turn)
        print('sheet: ',generated_sheet)
        
        generated_sequences.append([protein] + target_features + generated_features)
    
        print('\n \n \n')








        









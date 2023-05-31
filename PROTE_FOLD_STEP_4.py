# Zachary Webel
# zow2@georgetown.edu

'''
Step 4 creates data that can be analyzed by the network for training, val and test. 
'''
#########################################################################################


# Imports
import pandas as pd
from keras.utils import to_categorical
import numpy as np
from random import randint
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random



# import my embeddings
embedding_matrix = pd.read_csv('aa_embeddings.csv')
embedding_matrix.columns=['amino_acid','x1','x2','x3']

# import train and test seqs
train = pd.read_csv('train_database.csv')
validation = pd.read_csv('validation_database.csv')
test = pd.read_csv('test_database.csv')


# Define amio acid list, seq length and step size
amino_acid_list = list(embedding_matrix['amino_acid'])
seq_length = 10
step_size = 1


# Initialize storage
input_ = []
output = []
database = []
property_matrix = []



for sequence in train['sequence']:
    # random replacement if invalid token, then join to a string representation
    sequence_amino_acids = [aa if aa in amino_acid_list else random.choice(amino_acid_list) for aa in sequence]
    sequence = ''.join(sequence_amino_acids)
    
    # TARGET FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
    ###################################################################################
    analysed_seq = ProteinAnalysis(Seq(sequence))
    secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
    target_helix, target_turn, target_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
    target_features = [target_helix,target_turn,target_sheet]
    ###################################################################################

    # slide through (L to R) the sequence until there aren't any more 10 aa windows left
    for i in range(0,len(sequence_amino_acids) - seq_length, step_size):
        
        # create a list of the amino acids from i to i + 10 (10 amino acids input)
        input_sequence = sequence_amino_acids[i: i + seq_length]
        
        # i+10 position is the output amino acid
        output_amino_acid = sequence_amino_acids[i+seq_length]
        
        # Check if we have already analyzed this input/output pair
        if(''.join(input_sequence)+output_amino_acid not in database):
            database.append(''.join(input_sequence)+output_amino_acid)
            
            
            
            # WINDOW FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
            ###################################################################################
            analysed_seq = ProteinAnalysis(Seq("".join(input_sequence)))
            secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
            window_helix, window_turn, window_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
            window_features = [window_helix,window_turn,window_sheet]
            ###################################################################################
            
            
            
            # collect all amino acids in our current built sequence so far
            # every pass through the loop this will get bigger by one (adding on the right)
            current_built_sequence = sequence_amino_acids[0: i + seq_length]
            current_built_sequence = ''.join(current_built_sequence)
            
            # CURRENT_BUILT FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
            ###################################################################################
            analysed_seq = ProteinAnalysis(Seq(current_built_sequence))
            secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
            current_built_helix, current_built_turn, current_built_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
            current_built_features = [current_built_helix,current_built_turn,current_built_sheet]
            target_current_difference = [target_features[i]-current_built_features[i] for i in range(len(target_features))]
            ###################################################################################

            # for each amino acid in our input - translate to the embedding vector
            input.append([embedding_matrix.loc[embedding_matrix['amino_acid'] == aa].values[0][1:] for aa in input_sequence])

            # one hot encode the output amino acid based on the index from the embedding matrix
            # the last position corresponds to the stop tag if the output amino acid is the last in the sequence
            one_hot_output = [0]*21
            if(i+seq_length+1==len(sequence_amino_acids)):
                one_hot_output[-1] = 1
            else:
                index = embedding_matrix[embedding_matrix['amino_acid'] == output_amino_acid].index[0]
                one_hot_output[index] = 1
            output.append(one_hot_output)
            
            # concatenate the secondary structure features into one vector and append it to our storage vector
            property_matrix.append(target_features+window_features+target_current_difference)


# convert to a pandas data frame
property_matrix = pd.DataFrame(property_matrix)            
            
# reshape to a tensor size (len, 10, 3)
X = np.reshape(input[:(len(output))], (len(output), seq_length,3))
y = np.array(output)

# force floating point type for the embedding sequence
X = np.array(X).astype(float)

# force int type for one hot output 
y = np.array(y).astype(int)

# save
np.save('PROTE_FOLD_test_input.npy',X)
np.save('PROTE_FOLD_test_output.npy',y)
property_matrix.to_csv('PROTE_FOLD_test_property_matrix.csv',index=False)





            

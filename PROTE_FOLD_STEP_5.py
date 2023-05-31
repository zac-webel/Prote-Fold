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
    
    # Fit model
    model.fit([X_train,prop_train],y_train,
         epochs=1000,
         batch_size=32,
         validation_data=[[X_val,prop_val],y_val],
         callbacks=callbacks)


# Evaluate model
val_loss, val_accuracy = model.evaluate([X_val,prop_val],y_val)
print("Val Loss: ", val_loss)
print("Val Accuracy: ", val_accuracy)

test_loss, test_accuracy = model.evaluate([X_test,prop_test],y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)





# make reliability curve
predicted_distributions = model.predict([X_test,prop_test])
probability = []
result = []
# store the rouned probability and the binary result of next amino acid
for i in range(len(predicted_distributions)):
    for j in range(len(predicted_distributions[0])):
        probability.append(round(100*predicted_distributions[i][j],0))
        result.append(y_test[i][j])

# make a pandas df of the probs and binary result
import pandas as pd
test_predictions = pd.DataFrame()
test_predictions['probability'] = probability
test_predictions['result'] = result


x = []
y = []
cur = 0
# while loop through each valid probability 0-100
while cur<=1000:
    # if the model predicted an amino acid with that probability
    # store the probability and the actual frequency of that amino acid being a correct prediction
    if(len(test_predictions.loc[test_predictions.probability==cur])>100):
        x.append(cur)
        y.append(100*sum(test_predictions.loc[test_predictions.probability==cur]['result'])/len(test_predictions.loc[test_predictions.probability==cur])) 
    cur = cur + 1

# plot y=x and our threshold probabilities vs actual rate
import matplotlib.pyplot as plt
plt.scatter(x=x,y=y)
line = [x for x in range(0,100)]
plt.plot(line)





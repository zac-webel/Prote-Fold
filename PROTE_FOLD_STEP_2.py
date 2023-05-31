# Zachary Webel
# zow2@georgetown.edu

'''
Step 2 involves cleaning the database. Then Creating train,val and test sets. 
'''
#########################################################################################


# Imports
import pandas as pd


# Read in database
file_path = 'protein_database.csv'
database = pd.read_csv(file_path)


# Drop duplicates
database.drop_duplicates(subset=['sequence'], inplace=True)
database = database.reset_index(drop=True)


# Drop unwanted sequences - more cleaning can be done in the future
database = database[~database['description'].str.contains('assembly')].reset_index(drop=True)
database = database[~database['description'].str.contains('measure')].reset_index(drop=True)
database = database[~database['description'].str.contains('connector')].reset_index(drop=True)
database = database[~database['description'].str.contains('hypothetical')].reset_index(drop=True)
database = database[~database['description'].str.contains('collar')].reset_index(drop=True)
database = database[~database['description'].str.contains('Collar')].reset_index(drop=True)
database = database[~database['description'].str.contains('uncharacterized')].reset_index(drop=True)
database = database[~database['description'].str.contains('chaperone')].reset_index(drop=True)
database = database[~database['description'].str.contains('short')].reset_index(drop=True)
database = database[~database['description'].str.contains('ImpA family metalloprotease')].reset_index(drop=True)
database = database[~database['description'].str.contains('repeat')].reset_index(drop=True)


# Define minimum length sequence
seq_min_len = 30
database['length'] = [len(database.loc[i]['sequence']) for i in range(len(database))]
database = database.loc[database.length>seq_min_len].reset_index(drop=True)



# Create Train, Val and Test sets
validation = database[database['organism'].str.contains('Paraburkholderia')].reset_index(drop=True)
test = database[database['organism'].str.contains('Yersinia')].reset_index(drop=True)
train = database[~database['organism'].str.contains('Paraburkholderia|Yersinia')].reset_index(drop=True)


# Export data
train.to_csv('train_database.csv',index=False)
validation.to_csv('validation_database.csv',index=False)
test.to_csv('test_database.csv',index=False)

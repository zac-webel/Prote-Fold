# Zachary Webel
# zow2@georgetown.edu

'''
Step 3 creates our amino acid embeddings from the training set using word2vec. 
'''
#########################################################################################


# Imports
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Read in database
file_path = 'train_database.csv'
train = pd.read_csv(file_path)


# Split each amino acid by a space
for index, row in train.iterrows():
    train.at[index, 'sequence'] = ' '.join(list(row['sequence']))


# Tokenize the sequences treating an individual amino acid as a token (20 total)
train['tokens'] = train['sequence'].apply(lambda x: word_tokenize(x))


# Define the amino acid list
amino_acids = "ACDEFGHIKLMNPQRSTVWY"


# Create a tokens list to apply word2vec
tokens = list(train['tokens'])


# Define embedding size and window size
embedding_size = 3
window_size = 5


# Use word2vec to create amino acid embeddings
model = Word2Vec(tokens, vector_size=embedding_size, window=window_size, min_count=1, sg=1)


# Get the embeddings for each amino acid 
aa_embeddings = {}
for aa in amino_acids:
    if aa in model.wv.key_to_index:
        aa_embeddings[aa] = model.wv.get_vector(aa)


# create a dataframe of my embeddings and transpose the matrix
embeddings = pd.DataFrame(aa_embeddings).T


# some pca for fun
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1])

# Label each point with its corresponding amino acide
for i, aa in enumerate(list(embeddings.index)):
    ax.text(X_pca[i, 0]+0.01, X_pca[i, 1]+0.001, aa, fontsize=16)
    
plt.show()


# Format and save embedding matrix
embeddings.insert(0,'amino_acid',embeddings.index)
embeddings = embeddings.reset_index(drop=True)
embeddings.to_csv('aa_embeddings.csv',index=False)

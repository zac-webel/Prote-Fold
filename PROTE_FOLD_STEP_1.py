# Zachary Webel
# zow2@georgetown.edu

'''
Step 1 is to collect protein sequences. I use the entrez protein database
and defined a custom query: '(long tail fibre[All Fields] OR long tail fiber[All Fields]) AND phage[All Fields]'
to return long tail fiber sequences. I then export them into a csv file for later use.
'''
#########################################################################################


# Imports
from Bio import Entrez, SeqIO
import pandas as pd
import csv


# Set your email address
Entrez.email = 'zow2@georgetown.edu'

# Set the query and retrieve the total number of records
query = '(long tail fibre[All Fields] OR long tail fiber[All Fields]) AND phage[All Fields]'
handle = Entrez.esearch(db='protein', term=query, retmax=0)
record = Entrez.read(handle)
total_count = int(record['Count'])
print(total_count,'\n')

# Fetch the protein records in batches of 1000
batch_size = 1000
data = []
for start in range(0, total_count, batch_size):
    end = min(total_count, start + batch_size)
    handle = Entrez.esearch(db='protein', term=query, retstart=start, retmax=batch_size)
    record = Entrez.read(handle)
    id_list = record['IdList']
    handle = Entrez.efetch(db='protein', id=id_list, rettype='gb', retmode='text')
    records = SeqIO.parse(handle, 'genbank')
    for record in records:
        row = {
            'id': record.id,
            'description': record.description,
            'sequence': str(record.seq),
            'organism': record.annotations.get('organism'),
            'taxonomy': record.annotations.get('taxonomy')
        }
        data.append(row)
        lendata = len(data)
        # update progress, takes some time
        if(lendata%1000==0):
            print(lendata)

# Convert to data frame
df = pd.DataFrame(data)


# Save data
df.to_csv('protein_database.csv',index=False)

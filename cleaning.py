import pandas as pd
import nltk as nl
import re

clean_doc = pd.read_csv('Dataset/CompleteYoutubeDataset.csv')


def Ciao(ciao):
    ciao = re.sub("\s+", " ", ciao)
    return ciao


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(Ciao)
pipo = clean_doc['CONTENT']

for lines in pipo:
    print(lines)

clean_doc.to_csv('Dataset/DatasetNoSpace.csv')


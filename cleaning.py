import pandas as pd
import nltk as nl
import re

clean_doc = pd.read_csv('Dataset/CompleteYoutubeDataset.csv')


def spaces_remover(ciao):  #questa funzione rimuove spazi consecutivi > 1 dal dataset
    ciao = re.sub("\s+", " ", ciao)
    return ciao


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(spaces_remover)
clean = clean_doc['CONTENT']


def punt_remover(ciao): #questa funzione rimuove la punteggiatura dal dataset
    ciao = re.sub("[^-9A-Za-z]", "", ciao)
    return ciao


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(punt_remover)
clean = clean_doc['CONTENT']


def lower_converter(ciao): #questa funzione converte le lettere maiuscole in minuscole
    return ciao.lower()


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(lower_converter)
clean = clean_doc['CONTENT']


for lines in clean:
    print(lines)



clean_doc.to_csv('Dataset/DatasetNoSpace.csv')




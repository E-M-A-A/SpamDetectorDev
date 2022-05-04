import nltk
import pandas as pd
import re
from nltk.tokenize import TweetTokenizer

clean_doc = pd.read_csv('Dataset/CompleteYoutubeDataset.csv')


def lower_converter(lines): #questa funzione converte le lettere maiuscole in minuscole
    return lines.lower()


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(lower_converter)
clean = clean_doc['CONTENT']


def break_remover(lines):  #questa funzione rimuove gli invii
    lines = re.sub("<br>", " ", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(break_remover)
clean = clean_doc['CONTENT']


def punt_remover(lines): #questa funzione rimuove la punteggiatura dal dataset
    lines = re.sub("[^A-Za-z ]", "", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(punt_remover)
clean = clean_doc['CONTENT']


def href_remover(lines):  #questa funzione rimuove i link
    lines = re.sub("href*\w+", "", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(href_remover)
clean = clean_doc['CONTENT']


def http_remover(lines):  #questa funzione rimuove i link
    lines = re.sub("http*\w+", "", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(http_remover)
clean = clean_doc['CONTENT']


def spaces_remover(lines):  #questa funzione rimuove spazi consecutivi > 1 dal dataset
    lines = re.sub("\s+", " ", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(spaces_remover)
clean = clean_doc['CONTENT']


stopwords = nltk.corpus.stopwords.words('english')
took = TweetTokenizer()


def stopwords_remover(x): #funzione per la rimozione delle stopwords
    x = took.tokenize(x)
    post = [i for i in x if i not in stopwords]
    return " ".join(post)


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(stopwords_remover)
clean = clean_doc['CONTENT']
for lines in clean:
    print(lines)

clean_doc.to_csv('Dataset/DatasetNoSpace.csv')




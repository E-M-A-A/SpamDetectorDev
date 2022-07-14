import json

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from joblib import dump, load
"""

"""
clean_doc = pd.read_csv('Dataset/CompleteYoutubeDatasetCleanNoNullVal.csv')

colContenuto = clean_doc['CONTENT']
colEtichetta = clean_doc['CLASS']

cv = CountVectorizer()
X = cv.fit_transform(colContenuto)
X.toarray()

print(X)

contenutoAdd, contenutoTest, tagAdd, tagTest = train_test_split(X, colEtichetta, test_size=0.30, random_state=42) #splittiamo il dataset in 4 parti: la colonna contenente il contenuto dei commenti è diviso nella parte di training e di addestramento, e la colonna delle etichette dei commenti è divisa in parte training ed in parte test

agente = MultinomialNB() #restituisce un'istanza dell'algoritmo da utilizzare
agente.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente

print("Accuracy of Model",agente.score(contenutoTest,tagTest)*100,"%") #testiamo l'algoritmo
dump(agente, "fileJOBLIB/fileAgente.joblib") #così salviamo il nostro agente in un file a parte
dump(cv, "fileJOBLIB/dizionario.joblib") #così salviamo il nostro dizionario in un file a parte














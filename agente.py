import json

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
"""
import socket

HOST = "localhost"
PORT = 9997
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print("Connessione fallita. codice errore: " .format(err))
    s.close()
    exit(0)
print("Attendo connessioni")
s.listen()
conn, addr = s.accept()
print(f"Connesso da {addr}")
while True:
    data = conn.recv(1024)
    y = json.loads(data.decode("UTF-8"))
    for x in y:
        print(x["contenuto"])
    if not data:
        break
    print(data.decode("UTF-8"))
    s.close()
"""
clean_doc = pd.read_csv('Dataset/CompleteYoutubeDatasetCleanNoNullVal.csv')

colContenuto = clean_doc['CONTENT']
colEtichetta = clean_doc['CLASS']

cv = CountVectorizer()
X = cv.fit_transform(colContenuto)
X.toarray()

print(X)

contenutoAdd, contenutoTest, tagAdd, tagTest = train_test_split(X, colEtichetta, test_size=0.30, random_state=42) #splittiamo il dataset in 4 parti: la colonna contenente il contenuto dei commenti è diviso nella parte di training e di addestramento, e la colonna delle etichette dei commenti è divisa in parte training ed in parte test

instance = MultinomialNB() #restituisce un'istanza dell'algoritmo da utilizzare
instance.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente

print("Accuracy of Model",instance.score(contenutoTest,tagTest)*100,"%") #testiamo l'algoritmo






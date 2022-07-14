import json

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from joblib import dump, load
import socket

import cleanFun

agente = load("fileJOBLIB/fileAgente.joblib")  # ricarichiamo il nostro agente
cv = load("fileJOBLIB/dizionario.joblib")  # ricarichiamo la nostra libreria

HOST = "localhost"
PORT = 9998
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print("Connessione fallita. codice errore: ".format(err))
    s.close()
    exit(0)
print("Attendo connessioni")
s.listen()
conn, addr = s.accept()
print(f"Connesso da {addr}")
while True:
    data = conn.recv(4096)
    print(data)
    # y = json.loads()
    y = pd.read_json(data.decode("UTF-8"))
    y["contenuto"] = y["contenuto"] \
        .apply(cleanFun.lower_converter) \
        .apply(cleanFun.break_remover) \
        .apply(cleanFun.punt_remover) \
        .apply(cleanFun.href_remover) \
        .apply(cleanFun.http_remover) \
        .apply(cleanFun.spaces_remover) \
        .apply(cleanFun.correct_words) \
        .apply(cleanFun.stopwords_remover) \
        .apply(cleanFun.lemmatizer) \
        .apply(cleanFun.word_correction)

    vect = cv.transform(
        y["contenuto"]).toarray()  # vettorizziamo il commento da esaminare utilizzando la nostra libreria
    x = agente.predict(vect)  # facciamo eseguire la predizione all'agente
    print(x)
    names = []
    for i in range(len(x)):
        if x[i] == 1:
            print()
            names.append(y["username"][i])
    j = json.dumps(names) + '\n'
    print(j)
    print(j.encode("UTF-8"))
    conn.sendall(j.encode("UTF-8"))
    s.close()
    break

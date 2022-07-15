import concurrent.futures
import json
import pandas as pd
from joblib import load
import socket

import cleanFun


def correct(commenti, n):
    commenti = commenti \
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
    return commenti, n


agente = load("fileJOBLIB/fileAgente.joblib")  # ricarichiamo il nostro agente
cv = load("fileJOBLIB/dizionario.joblib")  # ricarichiamo la nostra libreria

NUMPROCESS = 4

HOST = "localhost"
PORT = 9999
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
    y = pd.read_json(data.decode("UTF-8"))
    futures = set()
    results = [None] * NUMPROCESS
    size = len(y["contenuto"])
    sizeEach = int(size / NUMPROCESS)
    r = size % NUMPROCESS
    if r > 0:
        results = [None] * (NUMPROCESS+1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUMPROCESS) as fut:
        for i in range(NUMPROCESS):
            x = fut.submit(correct, y["contenuto"].iloc[i * sizeEach:i * sizeEach + sizeEach], i)
            futures.add(x)
        for future in concurrent.futures.as_completed(futures):
            res, index = future.result()
            results[index] = res
    if r > 0:
        results[NUMPROCESS], i = correct(y["contenuto"].iloc[NUMPROCESS * sizeEach:NUMPROCESS * sizeEach + r],
                                         NUMPROCESS)
    y["contenuto"] = pd.concat(results)
    vect = cv.transform(
        y["contenuto"]).toarray()  # vettorizziamo il commento da esaminare utilizzando la nostra libreria
    x = agente.predict(vect)  # facciamo eseguire la predizione all'agente
    names = []
    print("Commenti trovati spam: ")
    for i in range(len(x)):
        if x[i] == 1:
            print(y['contenuto'][i])
            names.append(y["username"][i])
    j = json.dumps(names) + '\n'
    conn.sendall(j.encode("UTF-8"))
    s.close()
    break

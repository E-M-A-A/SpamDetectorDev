import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import re

from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.metrics import precision_score, PrecisionRecallDisplay
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm._libsvm import predict

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



agente = MultinomialNB() #restituisce un'istanza dell'algoritmo da utilizzare
agente.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente

#display = PrecisionRecallDisplay.766(
#    agente, contenutoTest, tagTest, name="LinearSVC"
#)
#_ = display.ax_.set_title("2-class Precision-Recall curve")

count = len(tagTest)

accuratezza = agente.score(contenutoTest,tagTest)*100

print("Accuracy of Model MultinomialNB",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)



agente2 = ComplementNB() #restituisce un'istanza dell'algoritmo da utilizzare
agente2.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente2.score(contenutoTest,tagTest)*100
print("Accuracy of Model ComplementNB",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)


agente3 = BernoulliNB() #restituisce un'istanza dell'algoritmo da utilizzare
agente3.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente3.score(contenutoTest,tagTest)*100
print("Accuracy of Model BernoulliNB ",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)



agente4 =  NearestCentroid() #restituisce un'istanza dell'algoritmo da utilizzare
agente4.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente4.score(contenutoTest,tagTest)*100
print("Accuracy of Model NearestCentroid",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)


agente5 =  SGDClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente5.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente5.score(contenutoTest,tagTest)*100
print("Accuracy of Model SGDClassifier",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)


agente6 =  LinearSVC() #restituisce un'istanza dell'algoritmo da utilizzare
agente6.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente6.score(contenutoTest,tagTest)*100
print("Accuracy of Model LinearSVC",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)



agente7 =  RandomForestClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente7.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente7.score(contenutoTest,tagTest)*100
print("Accuracy of Model RandomForestClassifier",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)


agente8 =  KNeighborsClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente8.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente8.score(contenutoTest,tagTest)*100
print("Accuracy of Model KNeighborsClassifier",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)


agente9 =  PassiveAggressiveClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente9.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente9.score(contenutoTest,tagTest)*100
print("Accuracy of Model PassiveAggressiveClassifier",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)



agente10 =  Perceptron() #restituisce un'istanza dell'algoritmo da utilizzare
agente10.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente10.score(contenutoTest,tagTest)*100
print("Accuracy of Model Perceptron",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)



agente11 =  RidgeClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente11.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza = agente11.score(contenutoTest,tagTest)*100
print("Accuracy of Model RidgeClassifier",accuratezza,"%") #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)



#agente12 =  Pipeline(5) #restituisce un'istanza dell'algoritmo da utilizzare
#agente12.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
#accuratezza = agente12.score(contenutoTest,tagTest)*100
#print("Accuracy of Model RidgeClassifier",accuratezza,"%") #testiamo l'algoritmo
#corrette = (count/100) * accuratezza
#print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)




#array = predict(contenutoTest, tagTest)
#print(array)
#prova = agente.precision_score(corrette, count)
#print("PROVAAAA", prova)


#agente.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

#dump(agente, "fileJOBLIB/fileAgente.joblib"); #così salviamo il nostro agente in un file a parte
#dump(cv, "fileJOBLIB/dizionario.joblib"); #così salviamo il nostro dizionario in un file a parte














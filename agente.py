import json

import numpy as np
from matplotlib import pyplot as plt
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
accuratezza1 = agente2.score(contenutoTest,tagTest)*100
print("Accuracy of Model ComplementNB",accuratezza1,"%") #testiamo l'algoritmo
corrette1 = (count/100) * accuratezza1
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette1)


agente3 = BernoulliNB() #restituisce un'istanza dell'algoritmo da utilizzare
agente3.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza2 = agente3.score(contenutoTest,tagTest)*100
print("Accuracy of Model BernoulliNB ",accuratezza2,"%") #testiamo l'algoritmo
corrette2 = (count/100) * accuratezza2
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette2)



agente4 =  NearestCentroid() #restituisce un'istanza dell'algoritmo da utilizzare
agente4.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza3 = agente4.score(contenutoTest,tagTest)*100
print("Accuracy of Model NearestCentroid",accuratezza3,"%") #testiamo l'algoritmo
corrette3 = (count/100) * accuratezza3
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette3)


agente5 =  SGDClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente5.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza4 = agente5.score(contenutoTest,tagTest)*100
print("Accuracy of Model SGDClassifier",accuratezza4,"%") #testiamo l'algoritmo
corrette4 = (count/100) * accuratezza4
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette4)


agente6 =  LinearSVC() #restituisce un'istanza dell'algoritmo da utilizzare
agente6.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza5 = agente6.score(contenutoTest,tagTest)*100
print("Accuracy of Model LinearSVC",accuratezza5,"%") #testiamo l'algoritmo
corrette5 = (count/100) * accuratezza5
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette5)



agente7 =  RandomForestClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente7.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza6 = agente7.score(contenutoTest,tagTest)*100
print("Accuracy of Model RandomForestClassifier",accuratezza6,"%") #testiamo l'algoritmo
corrette6 = (count/100) * accuratezza6
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette6)


agente8 =  KNeighborsClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente8.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza7 = agente8.score(contenutoTest,tagTest)*100
print("Accuracy of Model KNeighborsClassifier",accuratezza7,"%") #testiamo l'algoritmo
corrette7 = (count/100) * accuratezza7
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette7)


agente9 =  PassiveAggressiveClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente9.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza8 = agente9.score(contenutoTest,tagTest)*100
print("Accuracy of Model PassiveAggressiveClassifier",accuratezza8,"%") #testiamo l'algoritmo
corrette8 = (count/100) * accuratezza8
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette8)



agente10 =  Perceptron() #restituisce un'istanza dell'algoritmo da utilizzare
agente10.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza9 = agente10.score(contenutoTest,tagTest)*100
print("Accuracy of Model Perceptron",accuratezza9,"%") #testiamo l'algoritmo
corrette9 = (count/100) * accuratezza9
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette9)



agente11 =  RidgeClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
agente11.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
accuratezza10 = agente11.score(contenutoTest,tagTest)*100
print("Accuracy of Model RidgeClassifier",accuratezza10,"%") #testiamo l'algoritmo
corrette10 = (count/100) * accuratezza10
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette10)



#agente12 =  Pipeline(5) #restituisce un'istanza dell'algoritmo da utilizzare
#agente12.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
#accuratezza = agente12.score(contenutoTest,tagTest)*100
#print("Accuracy of Model RidgeClassifier",accuratezza,"%") #testiamo l'algoritmo
#corrette = (count/100) * accuratezza
#print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)




#array1 = predict(contenutoTest)
#print(array1)
#prova = agente.precision_score(corrette, count)
#print("PROVAAAA", prova)


#agente.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

#dump(agente, "fileJOBLIB/fileAgente.joblib"); #così salviamo il nostro agente in un file a parte
#dump(cv, "fileJOBLIB/dizionario.joblib"); #così salviamo il nostro dizionario in un file a parte





barWidth = 0.25
fig = plt.subplots(figsize=(15, 8))

# set height of bar
ACCURATEZZA = [accuratezza, accuratezza1, accuratezza2, accuratezza3, accuratezza4, accuratezza5, accuratezza6, accuratezza7, accuratezza8, accuratezza9, accuratezza10]
CORRETTE = [corrette, corrette1, corrette2, corrette3, corrette4, corrette5, corrette6, corrette7, corrette8, corrette9, corrette10]

# Set position of bar on X axis
br1 = np.arange(len(ACCURATEZZA))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.barh(br1, ACCURATEZZA, color='m', height=barWidth,
        edgecolor='black', label='ACCURATEZZA')
plt.barh(br2, CORRETTE, color='g', height=barWidth,
        edgecolor='black', label='ISTANZE CORRETTE')


# Adding Xticks
plt.xlabel('', fontweight='bold', fontsize=15)
plt.ylabel('', fontweight='bold', fontsize=15)
plt.yticks([r + barWidth for r in range(len(ACCURATEZZA))],
           ['MultinomialNB', 'ComplementNB', 'BernoulliNB', 'NearestCentroid', 'SGDClassifier', 'LinearSVC', 'RandomForestClassifier', 'KNeighborsClassifier', 'PassiveAggressiveClassifier', 'Perceptron', 'RidgeClassifierPerceptron'])

plt.legend()
plt.show()

wordcloud1 = WordCloud(width=800, height=500, margin=10, random_state=3, collocations=True).generate(' '.join(word_text))

plt.figure(figsize=(15,8))
    plt.imshow(wordcloud1, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    word_text=[]












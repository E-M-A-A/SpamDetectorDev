import json
import time

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


clean_doc = pd.read_csv('Dataset/CompleteYoutubeDatasetCleanNoNullVal.csv')

colContenuto = clean_doc['CONTENT']
colEtichetta = clean_doc['CLASS']



cv = CountVectorizer()
X = cv.fit_transform(colContenuto)
X.toarray()

print(X)

contenutoAdd, contenutoTest, tagAdd, tagTest = train_test_split(X, colEtichetta, test_size=0.30, random_state=42) #splittiamo il dataset in 4 parti: la colonna contenente il contenuto dei commenti è diviso nella parte di training e di addestramento, e la colonna delle etichette dei commenti è divisa in parte training ed in parte test





#display = PrecisionRecallDisplay.766(
#    agente, contenutoTest, tagTest, name="LinearSVC"
#)
#_ = display.ax_.set_title("2-class Precision-Recall curve")

count = len(tagTest)

agente = MultinomialNB() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA = f-s
s = time.time()
accuratezza = agente.score(contenutoTest,tagTest)*100
f = time.time()
tempoT = f-s
print("Accuracy of Model MultinomialNB",accuratezza,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette = (count/100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)

agente1 = ComplementNB() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente1.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA1 = f-s
s = time.time()
accuratezza1 = agente1.score(contenutoTest,tagTest)*100
f = time.time()
tempoT1 = f - s
print("Accuracy of Model ComplementNB",accuratezza1,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette1 = (count/100) * accuratezza1
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette1)

agente2 = BernoulliNB() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente2.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA2 = f-s
s = time.time()
accuratezza2 = agente2.score(contenutoTest,tagTest)*100
f = time.time()
tempoT2 = f - s
print("Accuracy of Model BernoulliNB ",accuratezza2,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette2 = (count/100) * accuratezza2
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette2)


agente3 =  NearestCentroid() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente3.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA3 = f-s
s = time.time()
accuratezza3 = agente3.score(contenutoTest,tagTest)*100
f = time.time()
tempoT3 = f - s
print("Accuracy of Model NearestCentroid",accuratezza3,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette3 = (count/100) * accuratezza3
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette3)

agente4 =  SGDClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente4.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA4 = f-s
s = time.time()
accuratezza4 = agente4.score(contenutoTest,tagTest)*100
f = time.time()
tempoT4 = f - s
print("Accuracy of Model SGDClassifier",accuratezza4,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette4 = (count/100) * accuratezza4
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette4)

agente5 =  LinearSVC() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente5.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA5 = f-s
s = time.time()
accuratezza5 = agente5.score(contenutoTest,tagTest)*100
f = time.time()
tempoT5 = f - s
print("Accuracy of Model LinearSVC",accuratezza5,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette5 = (count/100) * accuratezza5
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette5)

"""
agente6 =  RandomForestClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente6.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA6 = f-s
s = time.time()
accuratezza6 = agente6.score(contenutoTest,tagTest)*100
f = time.time()
tempoT6 = f - s
print("Accuracy of Model RandomForestClassifier",accuratezza6,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette6 = (count/100) * accuratezza6
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette6)


agente7 =  KNeighborsClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente7.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA7 = f-s
s = time.time()
accuratezza7 = agente7.score(contenutoTest,tagTest)*100
f = time.time()
tempoT7 = f - s
print("Accuracy of Model KNeighborsClassifier: ", accuratezza7, "%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette7 = (count/100) * accuratezza7
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette7)
"""

agente8 =  PassiveAggressiveClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente8.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA8 = f-s
s = time.time()
accuratezza8 = agente8.score(contenutoTest,tagTest)*100
f = time.time()
tempoT8 = f - s
print("Accuracy of Model PassiveAggressiveClassifier",accuratezza8,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette8 = (count/100) * accuratezza8
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette8)



agente9 =  Perceptron() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente9.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA9 = f-s
s = time.time()
accuratezza9 = agente9.score(contenutoTest,tagTest)*100
f = time.time()
tempoT9 = f - s
print("Accuracy of Model Perceptron",accuratezza9,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
corrette9 = (count/100) * accuratezza9
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette9)



agente10 =  RidgeClassifier() #restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente10.fit(contenutoAdd,tagAdd) #addestriamo il nostro agente intelligente
f = time.time()
tempoA10 = f-s
s = time.time()
accuratezza10 = agente10.score(contenutoTest,tagTest)*100
f = time.time()
tempoT10 = f - s
print("Accuracy of Model RidgeClassifier",accuratezza10,"%. Execution time: ", str(f-s)) #testiamo l'algoritmo
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
ACCURATEZZA = [accuratezza, accuratezza1, accuratezza2, accuratezza3, accuratezza4, accuratezza5, accuratezza8, accuratezza9, accuratezza10]
CORRETTE = [corrette, corrette1, corrette2, corrette3, corrette4, corrette5, corrette8, corrette9, corrette10]
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
           ['MultinomialNB', 'ComplementNB', 'BernoulliNB', 'NearestCentroid', 'SGDClassifier', 'LinearSVC', 'PassiveAggressiveClassifier', 'Perceptron', 'RidgeClassifierPerceptron'])

plt.legend()
plt.show()

TEMPOT = [tempoT * 1000, tempoT1 * 1000, tempoT2 * 1000, tempoT3 * 1000, tempoT4 * 1000, tempoT5 * 10000, tempoT8 * 1000, tempoT9 * 1000, tempoT10 * 1000]
TEMPOA = [tempoA * 1000, tempoA1 * 1000, tempoA2 * 1000, tempoA3 * 1000, tempoA4 * 1000, tempoA5 * 1000,  tempoA8 * 1000, tempoA9 * 1000, tempoA10 * 1000]
br1 = np.arange(len(TEMPOT))
br2 = [x + barWidth for x in br1]
plt.barh(br1, TEMPOT, color='c', height=barWidth,
        edgecolor='black', label='TEMPO DI TESTING(ms)')
plt.barh(br2, TEMPOA, color='r', height=barWidth,
        edgecolor='black', label='TEMPO DI ADDESTRAMENTO(ms)')
# Adding Xticks
plt.xlabel('', fontweight='bold', fontsize=15)
plt.ylabel('', fontweight='bold', fontsize=15)
plt.yticks([r + barWidth for r in range(len(TEMPOT))],
           ['MultinomialNB', 'ComplementNB', 'BernoulliNB', 'NearestCentroid', 'SGDClassifier', 'LinearSVC', 'PassiveAggressiveClassifier', 'Perceptron', 'RidgeClassifierPerceptron'])

plt.legend()
plt.show()













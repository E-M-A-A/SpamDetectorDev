import time

import numpy as np
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, RidgeClassifier
from sklearn.metrics import recall_score, precision_score, classification_report, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid

from sklearn.svm import LinearSVC

clean_doc = pd.read_csv('Dataset/CompleteYoutubeDatasetCleanNoNullVal.csv')

colContenuto = clean_doc['CONTENT']
colEtichetta = clean_doc['CLASS']

cv = CountVectorizer()
X = cv.fit_transform(colContenuto)
X.toarray()

print(X)

contenutoAdd, contenutoTest, tagAdd, tagTest = train_test_split(X, colEtichetta, test_size=0.30,
                                                                random_state=42)  # splittiamo il dataset in 4 parti:
# la colonna contenente il contenuto dei commenti è diviso nella parte di training e di addestramento, e la colonna
# delle etichette dei commenti è divisa in parte training ed in parte test

count = len(tagTest)

agente = MultinomialNB()  # restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente.fit(contenutoAdd, tagAdd)  # addestriamo il nostro agente intelligente
f = time.time()
tempoA = f - s
s = time.time()
accuratezza = agente.score(contenutoTest, tagTest) * 100 #accuratezza
f = time.time()
tempoT = f - s
recall = recall_score(tagTest, agente.predict(contenutoTest))
precision = precision_score(tagTest, agente.predict(contenutoTest))
mcc = matthews_corrcoef(tagTest, agente.predict(contenutoTest))
print(classification_report(tagTest,agente.predict(contenutoTest)))
print("Accuracy of Model MultinomialNB", accuratezza, "%. Execution time: ", str(f - s))  # testiamo l'algoritmo
corrette = (count / 100) * accuratezza
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette)

agente1 = ComplementNB()  # restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente1.fit(contenutoAdd, tagAdd)  # addestriamo il nostro agente intelligente
f = time.time()
tempoA1 = f - s
s = time.time()
accuratezza1 = agente1.score(contenutoTest, tagTest) * 100
f = time.time()
tempoT1 = f - s

recall1 = recall_score(tagTest, agente1.predict(contenutoTest))
precision1 = precision_score(tagTest, agente1.predict(contenutoTest))
mcc1 = matthews_corrcoef(tagTest, agente1.predict(contenutoTest))
print("Accuracy of Model ComplementNB", accuratezza1, "%. Execution time: ", str(f - s))  # testiamo l'algoritmo
corrette1 = (count / 100) * accuratezza1
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette1)

dump(agente1, "fileJOBLIB/fileAgente.joblib")  # così salviamo il nostro agente in un file a parte
dump(cv, "fileJOBLIB/dizionario.joblib")  # così

agente2 = BernoulliNB()  # restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
agente2.fit(contenutoAdd, tagAdd)  # addestriamo il nostro agente intelligente
f = time.time()
tempoA2 = f - s
s = time.time()
accuratezza2 = agente2.score(contenutoTest, tagTest) * 100
f = time.time()
tempoT2 = f - s
recall2 = recall_score(tagTest, agente2.predict(contenutoTest))
precision2 = precision_score(tagTest, agente2.predict(contenutoTest))
mcc2 = matthews_corrcoef(tagTest, agente2.predict(contenutoTest))
print("Accuracy of Model BernoulliNB ", accuratezza2, "%. Execution time: ", str(f - s))  # testiamo l'algoritmo
corrette2 = (count / 100) * accuratezza2
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette2)

agente3 = GaussianNB()  # restituisce un'istanza dell'algoritmo da utilizzare
s = time.time()
contenutoProva = contenutoAdd.toarray()
tagProva = tagAdd.array
agente3.fit(contenutoProva, tagProva) # addestriamo il nostro agente intelligente
f = time.time()
tempoA3 = f - s
s = time.time()
contenutoTestProva = contenutoTest.toarray()
tagTestProva = tagTest.array
accuratezza3 = agente3.score(contenutoTestProva, tagTestProva) * 100
f = time.time()
tempoT3 = f - s
recall3 = recall_score(tagTest, agente.predict(contenutoTestProva))
precision3 = precision_score(tagTest, agente.predict(contenutoTestProva))
mcc3 = matthews_corrcoef(tagTest, agente.predict(contenutoTestProva))
print("Accuracy of Model GaussianNB", accuratezza3, "%. Execution time: ", str(f - s))  # testiamo l'algoritmo
corrette3 = (count / 100) * accuratezza3
print("Numero istanze: ", count, " Numero istanze classificate correttamente: ", corrette3)

barWidth = 0.25
fig = plt.subplots(figsize=(15, 8))

# set height of bar
ACCURATEZZA = [accuratezza, accuratezza1, accuratezza2, accuratezza3]
CORRETTE = [corrette, corrette1, corrette2, corrette3]
RECALL = [recall* 100, recall1* 100, recall2* 100, recall3* 100]
MCC = [mcc* 100, mcc1* 100, mcc2* 100, mcc3* 100]
PRECISION = [precision* 100, precision1* 100, precision2* 100, precision3* 100]
ZERI = [5, 5, 5, 5]
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
           ['MultinomialNB', 'ComplementNB', 'BernoulliNB', 'GaussianNB'])

plt.legend()
plt.show()

TEMPOT = [tempoT * 1000, tempoT1 * 1000, tempoT2 * 1000, tempoT3 * 1000]
TEMPOA = [tempoA * 1000, tempoA1 * 1000, tempoA2 * 1000, tempoA3 * 1000]
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
           ['MultinomialNB', 'ComplementNB', 'BernoulliNB', 'GaussianNB'])

plt.legend()
plt.show()


br1 = np.arange(len(TEMPOT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br1]


plt.barh(br1, PRECISION, color='c', height=barWidth,
         edgecolor='black', label='PRECISION')
plt.barh(br5, ZERI, color='w', height=barWidth,
         edgecolor='pink', label='ZERI')
plt.barh(br2, RECALL, color='g', height=barWidth,
         edgecolor='black', label='RECALL')
plt.barh(br3, MCC, color='r', height=barWidth,
         edgecolor='black', label='MCC')
plt.barh(br4, ACCURATEZZA, color='m', height=barWidth,
         edgecolor='black', label='ACCURATEZZA')

# Adding Xticks
plt.xlabel('', fontweight='bold', fontsize=15)
plt.ylabel('', fontweight='bold', fontsize=15)
plt.yticks([r + barWidth for r in range(len(TEMPOT))],
           ['MultinomialNB', 'ComplementNB', 'BernoulliNB', 'GaussianNB'])

plt.legend()
plt.show()

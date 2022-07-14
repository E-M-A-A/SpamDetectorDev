import json

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from joblib import dump, load

agente = load("fileJOBLIB/fileAgente.joblib") #ricarichiamo il nostro agente
cv = load("fileJOBLIB/dizionario.joblib") #ricarichiamo la nostra libreria


comment = ["Check this out I will be giving 50% offer on your first purchase"] # TODO Ale qua si andrà a prendere il commento effettivo
vect = cv.transform(comment).toarray() #vettorizziamo il commento da esaminare utilizzando la nostra libreria
agente.predict(vect) #facciamo eseguire la predizione all'agente


























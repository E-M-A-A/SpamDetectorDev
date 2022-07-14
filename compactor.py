import csv

import pandas
import pandas as pd

# Classe DataFrame csvFile
csvPsy = pd.read_csv('Dataset/DatasetCleanYoutube01-Psy.csv')
print(csvPsy)

csvKatyPerry = pd.read_csv('Dataset/DatasetCleanYoutube02-KatyPerry.csv')
print(csvKatyPerry)
csvLMFAO = pd.read_csv('Dataset/DatasetCleanYoutube03-LMFAO.csv')
csvEminem = pd.read_csv('Dataset/DatasetCleanYoutube04-Eminem.csv')
csvShakira = pd.read_csv('Dataset/DatasetCleanYoutube05-Shakira.csv')
#Array che contiene tutti i csv
cvsses = [csvPsy, csvKatyPerry, csvEminem, csvLMFAO, csvShakira]
#Concatenazione di tutti i csv
bigCsv = pd.concat(cvsses)
bigCsv = bigCsv.loc[:, ~bigCsv.columns.str.contains('^Unnamed')]
print(bigCsv)
#Scrittura su file csv
bigCsv.to_csv('Dataset/CompleteYoutubeDatasetClean.csv')


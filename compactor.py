import csv

import pandas
import pandas as pd

# Classe DataFrame csvFile
csvPsy = pd.read_csv('Dataset/Youtube01-Psy.csv')
print(csvPsy)

csvKatyPerry = pd.read_csv('Dataset/Youtube02-KatyPerry.csv')
print(csvKatyPerry)
csvLMFAO = pd.read_csv('Dataset/Youtube03-LMFAO.csv')
csvEminem = pd.read_csv('Dataset/Youtube04-Eminem.csv')
csvShakira = pd.read_csv('Dataset/Youtube05-Shakira.csv')
#Array che contiene tutti i csv
cvsses = [csvPsy, csvKatyPerry, csvEminem, csvLMFAO, csvShakira]
#Concatenazione di tutti i csv
bigCsv = pd.concat(cvsses)
print(bigCsv)
#Scrittura su file csv
bigCsv.to_csv('CompleteYoutubeDataset.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

docEm = pd.read_csv('Dataset/Youtube04-Eminem.csv')["CLASS"]
docKat = pd.read_csv('Dataset/Youtube02-KatyPerry.csv')["CLASS"]
docPsy = pd.read_csv('Dataset/Youtube01-Psy.csv')["CLASS"]
docL = pd.read_csv('Dataset/Youtube03-LMFAO.csv')["CLASS"]
docSha = pd.read_csv('Dataset/Youtube05-Shakira.csv')["CLASS"]

spamEm = 0
for i in docEm:
    if i == 1:
        spamEm += 1
hamEm = len(docEm) - spamEm

spamKat = 0
for i in docKat:
    if i == 1:
        spamKat += 1
hamKat = len(docKat) - spamKat

spamPsy = 0
for i in docPsy:
    if i == 1:
        spamPsy += 1
hamPsy = len(docPsy) - spamPsy

spamL = 0
for i in docL:
    if i == 1:
        spamL += 1
hamL = len(docL) - spamL

spamSha = 0
for i in docSha:
    if i == 1:
        spamSha += 1
hamSha = len(docSha) - spamSha

spamAll = spamSha + spamL + spamEm + spamPsy + spamKat
hamAll = hamSha + hamL + hamEm + hamPsy + hamKat
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar
SPAM = [spamSha, spamL, spamKat, spamPsy, spamEm]
HAM = [hamSha, hamL, hamKat, hamPsy, hamEm]

# Set position of bar on X axis
br1 = np.arange(len(SPAM))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, SPAM, color='m', width=barWidth,
        edgecolor='black', label='SPAM')
plt.bar(br2, HAM, color='g', width=barWidth,
        edgecolor='black', label='HAM')

# Adding Xticks
plt.xlabel('Dataset', fontweight='bold', fontsize=15)
plt.ylabel('No. Commenti', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(SPAM))],
           ['Shakira', 'LMFAO', 'KatyPerry', 'Psy', 'Eminem'])

plt.legend()
plt.show()

x = ["SPAM", "HAM"]
h = [spamAll, hamAll]
c = ['purple', 'green']

# bar plot
plt.figure(figsize=(6, 5))
plt.bar(x, height=h, color=c, width=0.4)
plt.xlabel('Tipo', fontweight='bold', fontsize=15)
plt.ylabel('No. Commenti', fontweight='bold', fontsize=15)

plt.show()

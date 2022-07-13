import pandas as pd
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from Tools.scripts.dutree import display

data = pd.read_cvs('Dataset/CompleteYoutubeDataset.csv')


#X = data.iloc[:, :-1]
#y = data.classified

#print('Data Table \n')
#display(X)
#print('\n\nTags Table')
#display(y)
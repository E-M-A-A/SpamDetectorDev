import pandas as pd

clean_doc = pd.read_csv('Dataset/CompleteYoutubeDatasetClean.csv')
print(clean_doc.isnull().count(axis=0))
clean_doc = clean_doc.dropna(axis=0, how='any')
clean_doc.to_csv('Dataset/CompleteYoutubeDatasetCleanNoNullVal.csv')

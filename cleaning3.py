import pandas as pd

import cleanFun

clean_doc = pd.read_csv('Dataset/Youtube03-LMFAO.csv')
# nome colonna,0 per la riga/1 per la colonna
clean_doc = clean_doc.drop("COMMENT_ID", 1)
clean_doc = clean_doc.drop("DATE", 1)
clean_doc = clean_doc.drop("AUTHOR", 1)

clean_doc["CONTENT"] = clean_doc["CONTENT"]\
    .apply(cleanFun.lower_converter) \
    .apply(cleanFun.break_remover) \
    .apply(cleanFun.punt_remover) \
    .apply(cleanFun.href_remover)\
    .apply(cleanFun.http_remover)\
    .apply(cleanFun.spaces_remover) \
    .apply(cleanFun.correct_words)\
    .apply(cleanFun.stopwords_remover)\
    .apply(cleanFun.lemmatizer) \
    .apply(cleanFun.word_correction)

clean_doc.to_csv('Dataset/DatasetCleanYoutube03-LMFAO.csv')

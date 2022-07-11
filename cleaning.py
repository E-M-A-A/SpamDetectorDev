import nltk
import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import words
from nltk.metrics.distance  import edit_distance

#TODO implementare un nuovo stemmer

clean_doc = pd.read_csv('Dataset/CompleteYoutubeDataset.csv')
#nome colonna,0 per la riga/1 per la colonna
clean_doc = clean_doc.drop("CONTATORE",1)
clean_doc = clean_doc.drop("COMMENT_ID",1)
clean_doc = clean_doc.drop("DATE",1)
clean_doc = clean_doc.drop("AUTHOR",1)





def lower_converter(lines): #questa funzione converte le lettere maiuscole in minuscole
    return lines.lower()


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(lower_converter)
clean = clean_doc['CONTENT']


def break_remover(lines):  #questa funzione rimuove gli invii
    lines = re.sub("<br>", " ", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(break_remover)
clean = clean_doc['CONTENT']


def punt_remover(lines): #questa funzione rimuove la punteggiatura dal dataset
    lines = re.sub("[^A-Za-z ]", "", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(punt_remover)
clean = clean_doc['CONTENT']


def href_remover(lines):  #questa funzione rimuove i link
    lines = re.sub("href*\w+", "", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(href_remover)
clean = clean_doc['CONTENT']


def http_remover(lines):  #questa funzione rimuove i link
    lines = re.sub("http*\w+", "", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(http_remover)
clean = clean_doc['CONTENT']


def spaces_remover(lines):  #questa funzione rimuove spazi consecutivi > 1 dal dataset
    lines = re.sub("\s+", " ", lines)
    return lines


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(spaces_remover)
clean = clean_doc['CONTENT']
nltk.download('words')
correct_words = words.words()
took = TweetTokenizer()


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def correct_words(text):
    line = took.tokenize(text)
    for word in line:
        print("checking", word)
        word = reduce_lengthening(word)
        print("correcting", word)

    return " ".join(line)


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(correct_words)
clean = clean_doc['CONTENT']


"""
def word_corrector(x):  #questa funzione rimuove spazi consecutivi > 1 dal dataset
    x = took.tokenize(x)
    for y in x:
        word_correction(y)
        print("corrected", y)
    return " ".join(x)


def word_correction(y):
    print("checking", y)
    if y in correct_words:
        return y
    temp = [(edit_distance(y, w), w) for w in correct_words if w[0] == y[0]]
    return sorted(temp, key=lambda val: val[0])[0][1]

clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(word_corrector)
clean = clean_doc['CONTENT']"""


stopwords = nltk.corpus.stopwords.words('english')


def stopwords_remover(x): #funzione per la rimozione delle stopwords
    x = took.tokenize(x)
    post = [i for i in x if i not in stopwords]
    return " ".join(post)


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(stopwords_remover)
clean = clean_doc['CONTENT']







#clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(stemmer)
#clean = clean_doc['CONTENT']

nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()


def lemmatizer(x): #funzione per la rimozione delle stopwords
    x = took.tokenize(x)
    w = [wn.lemmatize(i) for i in x]

    return " ".join(w)


clean_doc['CONTENT'] = clean_doc['CONTENT'].apply(lemmatizer)
clean = clean_doc['CONTENT']

    
for lines in clean:
    print(lines)

clean_doc.to_csv('Dataset/DatasetClean.csv')




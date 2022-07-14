import string
import random
import pandas as pd
import mysql.connector

import cleanFun

mydb = mysql.connector.connect(
    host='localhost',
    user='Storytelling',
    password='Ciao.123',
    database='storytelling'
)

mycursor = mydb.cursor()


def get_random_string():
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for p in range(7))
    return result_str


def get_users():
    sql = "SELECT username FROM utente WHERE password='Pippo.123';"
    mycursor.execute(sql)
    utenti = mycursor.fetchall()
    #print(utenti)
    users = []
    for u in utenti:
        users.append(u[0])
    return users


def get_stories():
    sql = "SELECT MAX(id) FROM storia;"
    mycursor.execute(sql)
    storie = mycursor.fetchone()
    #print(storie[0])
    return storie[0]


def fill_users():
    sql = 'INSERT INTO utente (email,username,password) VALUES (%s,%s,%s)'
    x = get_random_string()
    val = (x + '@asso.it', x, "Pippo.123")
    mycursor.execute(sql, val)
    mydb.commit()


def fill_comments(contenuto):
    sql = 'INSERT INTO commento (contenuto, username, idStoria) VALUES (%s, %s, %s)'
    users = get_users()
    maxidstories = get_stories()
    val = (contenuto, users[random.randrange(0, len(users))], random.randrange(1, maxidstories))
    mycursor.execute(sql, val)
    mydb.commit()


# for i in range(5):
#    fill_users()
# get_users()
"""
comments = pd.read_csv("/home/ale/spam_or_ham.csv")
commenti = comments["CONTENT"]
for i in range(10):
    if len(commenti[i]) > 128:
        continue
    fill_comments(commenti[i])
"""

# commenti = comments["CONTENT"].apply(cleanFun.word_correction) \
#    .apply(cleanFun.break_remover) \
#    .apply(cleanFun.spaces_remover) \
#    .apply(cleanFun.punt_remover) \
#    .apply(cleanFun.lower_converter) \
#    .apply(cleanFun.correct_words) \
#    .apply(cleanFun.lemmatizer) \
#    .apply(cleanFun.stopwords_remover)

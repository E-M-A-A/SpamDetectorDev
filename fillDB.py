import string
import random
import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(
    host='localhost',
    user='Storytelling',
    password='Ciao.123',
    database='storytelling'
)

mycursor = mydb.cursor()


# Generazione casuale di stringhe
def get_random_string():
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for p in range(7))
    return result_str


def get_users():
    sql = "SELECT username FROM utente WHERE password='Pippo.123'"
    mycursor.execute(sql)
    utenti = mycursor.fetchall()
    users = []
    for u in utenti:
        users.append(u[0])
    return users


def get_stories():
    sql = "SELECT id FROM storia"
    mycursor.execute(sql)
    storie = mycursor.fetchall()
    stories = []
    for s in storie:
        stories.append(s[0])
    return stories


# Generazione casuale di utenti
def fill_users():
    sql = 'INSERT INTO utente (email,username,password) VALUES (%s,%s,%s)'
    x = get_random_string()
    val = (x + '@asso.it', x, "Pippo.123")
    mycursor.execute(sql, val)
    mydb.commit()


# Inserimento commenti in maniera casuale
def fill_comments(contenuto):
    sql = 'INSERT INTO commento (contenuto, username, idStoria) VALUES (%s, %s, %s)'
    users = get_users()
    stories = get_stories()
    val = (contenuto, users[random.randrange(0, len(users))], stories[random.randrange(0, len(stories))])
    mycursor.execute(sql, val)
    mydb.commit()


for i in range(10):
    fill_users()

# Raccolta commenti da una data source
comments = pd.read_csv("~/spam_or_ham.csv")
commenti = comments["CONTENT"]
for i in range(30):
    if len(commenti[i]) > 128:
        continue
    fill_comments(commenti[i])

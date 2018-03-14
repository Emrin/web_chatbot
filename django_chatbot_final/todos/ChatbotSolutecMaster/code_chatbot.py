# -*- coding: utf-8 -*-
### Installation commands
import pip
pip.main(["install","tensorflow"])
pip.main(["install","tflearn"])
pip.main(["install","nltk"])
pip.main(["install","django"])
# Importation des libraries
import tensorflow as tf
import random
import numpy as np
import tflearn
import nltk
import json
import pickle
from tkinter import *

nltk.download("punkt")

# On met le stem en francais
stemmer = nltk.stem.SnowballStemmer('french')

# Importation des fichiers json pour l'ia
import os
curPath1 = os.path.dirname(os.path.abspath(__file__))
curPath2 = os.path.dirname(os.path.abspath(__file__))
curPath1 += '\\intents.json'
curPath2 += '\\salons.json'

with open(curPath1, encoding='utf8') as json_data:
    intents = json.load(json_data)
with open(curPath2, encoding='utf8') as json_salons:
    salons = json.load(json_salons)

words = []
classes = []
documents = []
ignore_words = ['?']
# On parcours les phrases
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # on divise la phrase en mots
        w = nltk.word_tokenize(pattern)
        # on ajoute les mots a la liste
        words.extend(w)
        # on joute les mots au document avec le tag correspondant
        documents.append((w, intent['tag']))
        # ajoute les tags a la liste de classe
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

name_meeting = []
date_meeting = []
city_meeting = []
duration_meeting = []
# On remplis les differents champ pour les salons
for salon in salons:
    name_meeting.append(salon['name'])
    date_meeting.append(salon['date_debut'])
    city_meeting.append(salon['ville'])
    duration_meeting.append(salon['duree'])
    
# On trie et on normalise les mots on enleve les doublons
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# On enlever les doublons
# Trie dans l'ordre alphabetique
classes = sorted(list(set(classes)))

# Affichages des differentes classe
print (len(documents), "documents", documents)
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


print("Initialisation de l'entrainement")
# Création de l'entrainement
training = []
output = []

# création d'un tableau vide
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Remplissage de training avec des données  aléatoire
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
tf.reset_default_graph()
# Construction du reseau neuronal
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# Utilisation de la fonction softmax
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Definition du modele et dinition du dossier d'enregistrement
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Début de l'entrainement
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

def normalisation_sentence(sentence):
    # permet de decouper la phrase en mot
    sentence_words = nltk.word_tokenize(sentence)
    # met tout les mots en minuscule puis les normalise avec le stemmer
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# retourne un tableau avec des 0 et des 1 en fonction de l'importance des mots dans la phrase
def bow(sentence, words):
    # cette fonction permet de normaliser la phrase
    sentence_words = normalisation_sentence(sentence)
    # on creer un tableau avec un 0 pour chaque mot
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            #si le mot est connu on lui donne de l'importance
            if w == s:
                bag[i] = 1
    return(np.array(bag))

#Test sur console
p = bow("Ou puis-je renconter Solutec ?", words)
mdl_pred = model.predict([p])
print (p)
print("class")
print (classes)
print("model prediction")
print(mdl_pred)

# sauvgarde de toute les données
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
context = {}

ERROR_THRESHOLD = 0.25
#trouve la bonne phrases selon le modele etablit par l'entrainement
def classify(sentence):
    #trouve la probabilité de chaque theme selon modele
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # retourne un tuple avec le theme et la porbabilité
    print(return_list)
    return return_list

import re
from datetime import datetime
#Permet d'ajouter les informations sur les salons aux phrases qui en parlent
def add_meeting(sentence):
    #On cherche prochain salon
    index_ref = 0
    for idx, val in enumerate(date_meeting):
        val_date = re.split('/', val)
        duration_gap  = datetime(int(val_date[2]), int(val_date[1]), int(val_date[0])) - datetime.now()
        if duration_gap.days >= 0 and duration_gap.days < 90 :
            index_ref = idx
    #On rpends les informations necessaire      
    meeting_begening = date_meeting[index_ref]
    meeting_duration =   duration_meeting[index_ref]
    meeting_name =     name_meeting[index_ref]
    meeting_city = city_meeting[index_ref]
    sentence = sentence.replace('&', meeting_city).replace('{', meeting_begening).replace('}', meeting_duration).replace('@',meeting_name)
    print (sentence)
    return (sentence)

#Choix de la réponse
def response(sentence, userID):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                # Cherche le theme correspondant la prediction du modele
                if i['tag'] == results[0][0]:                        
                    if 'context_set' in i:
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        # Choisit une reponse au hasard pammis celle proposées pour le sujet choisi
                        sentence_response = random.choice(i['responses'])
                        if i['tag'] == 'Rencontres':
                            sentence_response = add_meeting(sentence_response)
                        return (sentence_response)
            results.pop(0)

class Application(Frame):
    def send(self, *args, **kwargs):
        # saved_args = locals()
        # print("saved_args is", saved_args)
        if self.ENTRY.get() != "":
            self.TEXT.config(state=NORMAL)
            # Paste user entry
            user_input = "Vous : " + self.ENTRY.get() + "\r\n"
            self.TEXT.insert(END, user_input)
            print(self.ENTRY.get())

            # Now paste chatbot response
            bot_response = "Solutec : " + response(self.ENTRY.get(), '123', False) + "\r\n"
            self.TEXT.insert(END, bot_response)

            # Update some settings
            self.TEXT.config(state=DISABLED)
            self.TEXT.see("end")
            self.ENTRY.delete(0, "end")
        else:
            print("Empty entry")

    def createWidgets(self):
        self.TEXT = Text(self, bg="#bbd1f9", font="Arial 12 bold")
        self.TEXT.insert(END, "Solutec : Comment puis-je vous etre utile ?\r\n")
        self.TEXT.config(state=DISABLED)
        self.TEXT.grid(row=0, column=0, columnspan=2, sticky=N+S+E+W)

        # Scrollbar
        # self.SCROLL = Scrollbar(self, command=self.TEXT)

        self.ENTRY = Entry(self, bg="#eaf7ec", font="Arial 12 bold", highlightcolor="#80aef7", width=70,
                           highlightthickness="1", relief="groove")
        self.ENTRY.grid(row=1, column=0, sticky=W)
        self.ENTRY.focus()

        self.SEND = Button(self, text="Send", font="System", fg="White", bg="#1dd138", width="15",
                           command=self.send)
        self.SEND.grid(row=1, column=1, sticky=E)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()
        self.createWidgets()
        self.bind_all("<Return>", self.send)


#root = Tk()
#root.title("Chatbot SOLUTEC")
#oot.geometry("500x500")
#app = Application(master=root)
#app.mainloop()
#root.destroy()
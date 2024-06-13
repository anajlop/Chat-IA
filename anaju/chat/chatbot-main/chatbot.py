import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import tkinter as tk
from tkinter import scrolledtext

lemmatizer = WordNetLemmatizer()

# Importar los archivos generados en el código anterior
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Pasar las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convertir la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecir la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    predicted_class_index = np.argmax(result)
    predicted_class = classes[predicted_class_index]
    return predicted_class

# Obtener una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def respuesta(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res

# Función para enviar mensaje en la interfaz
def send_message():
    user_message = user_entry.get()
    chat_window.insert(tk.END, "You: " + user_message + "\n")
    chat_window.yview(tk.END)
    user_entry.delete(0, tk.END)

    bot_response = respuesta(user_message)
    chat_window.insert(tk.END, "Bot: " + bot_response + "\n")
    chat_window.yview(tk.END)

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Chatbot")

# Crear el cuadro de texto para la conversación
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, font=("Arial", 12))
chat_window.pack(padx=10, pady=10)

# Crear la entrada de texto para el usuario
user_entry = tk.Entry(root, width=40, font=("Arial", 12))
user_entry.pack(padx=10, pady=5)
user_entry.bind("<Return>", lambda event: send_message())

# Crear el botón de enviar
send_button = tk.Button(root, text="Enviar", command=send_message, font=("Arial", 12))
send_button.pack(padx=10, pady=5)

root.mainloop()



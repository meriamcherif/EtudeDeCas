from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from spacy.lang.en import STOP_WORDS
from nltk.corpus import wordnet
from transformers import pipeline
import nltk
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes
categories_keywords = {
        'Food': ['food', 'meal', 'dish', 'cuisine', 'flavor','cake','dessert','appetizer','beverage','pizza','pasta'],
        'Service': ['service', 'waiter', 'staff', 'friendly','attitude','hospitality',''],
        'Ambiance': ['ambiance', 'atmosphere', 'decor', 'music','setting','lighting',],
        'Price/Quality': ['price', 'value', 'money','affordability','cost'],
        'Overall Experience': ['experience', 'overall','dining','everything','nothing'],
        'Cleanliness': ['cleanliness', 'hygiene','tidiness','sanitation','neatness']
    }

# Charger le modèle spaCy avec embeddings de mots en anglais
nlp = spacy.load('en_core_web_md')
reponse=''
# Charger les données
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Fonction pour prétraiter le texte avec spaCy
def preprocess_text(text):
    doc = nlp(text)
    return np.mean([token.vector for token in doc], axis=0)

# Appliquer le prétraitement aux avis
df['processed_review'] = df['Review'].apply(preprocess_text)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(np.vstack(df['processed_review']), df['Liked'], test_size=0.2, random_state=42)

# Encoder les labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Construire le modèle de réseau de neurones
model = Sequential()
model.add(Dense(128, input_dim=300, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.1)

# Fonction pour prédire la polarité d'un nouvel avis
def predict_polarity(new_review, threshold=0.5):
    processed_new_review = preprocess_text(new_review)
    new_review_embedding = processed_new_review.reshape(1, -1)
    prediction_prob = model.predict(new_review_embedding)[0, 0]
    predicted_label = 1 if prediction_prob >= threshold else 0
    return prediction_prob, predicted_label

# Fonction pour diviser un avis en phrases et prédire la polarité de chaque phrase
def predict_polarity_per_sentence(review, threshold=0.5):
    sentences = nltk.sent_tokenize(review)
    polarities = []
    negative_sentences = []
    positive_sentences = []
    for sentence in sentences:
        prob, label = predict_polarity(sentence)
        polarities.append(label)
        if label == 0:
            negative_sentences.append(sentence)
        else:
            positive_sentences.append(sentence)
    total_polarity = "negative" if 0 in polarities else "positive"
    return total_polarity, negative_sentences, positive_sentences

# Fonction pour obtenir les synonymes d'un mot
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Fonction pour extraire les mots-clés et leurs synonymes d'un avis
def extract_keywords_with_synonyms(review):
    keywords = keyword_extractor(review)
    keywords_with_synonyms = []
    for keyword in keywords:
        keyword_with_synonyms = [keyword]
        synonyms = get_synonyms(keyword)
        keyword_with_synonyms += synonyms
        keywords_with_synonyms += keyword_with_synonyms
    return keywords_with_synonyms

def keyword_extractor(avis):
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy
    doc = nlp(avis)

    # Extract lemmatized words excluding stop words
    keywords = [token.lemma_ for token in doc if token.text.lower() not in STOP_WORDS and token.text.lower() not in ' ']

    # Display the extracted keywords
    return keywords

# Fonction pour trouver la catégorie principale d'un avis
def find_category(review):
    keywords_synonyms = []
    keywords = keyword_extractor(review)
    for keyword in keywords:
        synonyms = get_synonyms(keyword)
        keywords_synonyms.append(synonyms)
    max_similarity = 0
    best_category = None
    for category, category_keywords in categories_keywords.items():
        similarity = len(set(keywords) & set(category_keywords)) / len(set(keywords) | set(category_keywords))
        if similarity > max_similarity:
            max_similarity = similarity
            best_category = category
    if best_category is None:
        best_category = 'Overall Experience'
    return best_category

# Fonction pour déterminer le sentiment d'un avis
def feeling(review):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    result = classifier(review)
    max_score_label = max(result[0], key=lambda x: x['score'])['label']
    return max_score_label

# Fonction pour analyser un avis
def review_analysis_pipeline(review):
    total_polarity, negative_sentences, positive_sentences = predict_polarity_per_sentence(review)
    response=""
    if total_polarity == "positive":
        feel = feeling(review)
        categories = []
        for sentence in positive_sentences:
            category = find_category(sentence)
            categories.append(category)
        if len(categories) == 1:
            if categories[0] == "Food":
                response = "Thank you for your positive review! Your " + feel + " matters! We are happy that you enjoyed our food! Looking forward to serving you again."
            elif categories[0] == "Service":
                response = "Thank you for your positive review! Thank you for your kind words about our service! We appreciate your positive feedback and your " + feel + " does really matter! We look forward to serving you again."
            elif categories[0] == "Ambiance":
                response = "Thank you for your positive review! We're thrilled you loved the ambiance! Your " + feel + " means a lot to us. Looking forward to your next visit."
            elif categories[0] == "Price/Quality":
                response = "Hello and thank you for your positive feedback on the value we provide! We're delighted to hear you found our offerings worth the price. Your " + feel + " is our priority, and we can't wait to welcome you back for another great experience."
            elif categories[0] == "Overall Experience":
                response = "Hello! We're thrilled to hear you had a positive overall experience with us. Your " + feel + " is our priority, and we can't wait to welcome you back for another great visit. Thank you for your kind words!"
            elif categories[0] == "Cleanliness":
                response = "Hello and thank you for your positive feedback on the cleanliness of our establishment!"
        elif len(categories) == 2:
                response = "Thank you for your positive review! We're delighted that you appreciated our " + categories[0] + " and " + categories[1] + ". Looking forward to serving you again."

    elif total_polarity == "negative":
        feel = feeling(review)
        categories = []
        for sentence in negative_sentences:
            category = find_category(sentence)
            categories.append(category)
            if len(categories) == 1:
                response = "Greetings! We're sorry to hear about your experience with our " + categories[0] + ". Your feedback is valuable, and we'd like to address any concerns you have. Please let us know how we can enhance your experience next time."
            elif len(categories) == 2:
                response = "Greetings! We are sorry to hear about your experience with our " + categories[0] + " and " + categories[1] + ". Your feedback is valuable, and we'd like to address any concerns you have. Please let us know how we can enhance your experience next time."

    return response
# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html', response='')

# Route pour l'analyse du review
@app.route('/analyze', methods=['POST'])
def analyze():
    # Récupérer le review depuis le formulaire
    user_review = request.form['review']

    # Analyser le review et obtenir la réponse
    global response  # Utiliser global pour indiquer que nous faisons référence à la variable globale
    response = review_analysis_pipeline(user_review)

    # Retourner la page d'accueil avec la réponse
    return render_template('index.html', response=response)

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)



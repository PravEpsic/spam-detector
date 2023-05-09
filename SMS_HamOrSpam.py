import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Chargement des données depuis le fichier CSV
data = pd.read_csv('HamOrSpam.csv', encoding='UTF-8')

# Séparation des données en données d'entraînement et données de test
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], random_state=0)

# Vectorisation des données textuelles en utilisant TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Entrainement du modèle
model = LinearSVC()
model.fit(X_train_vect, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test_vect)

# Evaluation des performances du modèle
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Confusion Matrix:\n', conf_matrix)

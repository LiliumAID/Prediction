import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Učitavanje i čišćenje podataka
df = pd.read_csv("products.csv")
df = df.dropna(subset=['Product Title', 'Category Label'])

# Kodiranje kategorija
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category Label'])

# Podela podataka
X_train, X_test, y_train, y_test = train_test_split(
    df['Product Title'], df['Category_encoded'], test_size=0.2, random_state=42, stratify=df['Category_encoded']
)

# TF-IDF vektorizacija
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Treniranje Random Forest modela
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_tfidf, y_train)

# Čuvanje modela i transformera
pickle.dump(model, open('product_category_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("Model, vektorizer i enkoder uspešno sačuvani!")

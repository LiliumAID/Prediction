# Prediction
Ovaj projekat implementira model mašinskog učenja koji automatski predviđa kategoriju proizvoda na osnovu naziva proizvoda.
Cilj je ubrzati i automatizovati proces klasifikacije proizvoda u online trgovini, smanjiti greške i unaprediti korisničko iskustvo.

Model koristi TF-IDF vektorizaciju naslova proizvoda i Naive Bayes Classifier za predikciju kategorije.
Projekat uključuje kompletan pipeline:

Učitavanje i pregled podataka

Čišćenje i standardizacija podataka

Feature engineering (dužina naziva, broj reči, prisustvo brojeva)

Vektorizacija teksta sa TF-IDF

Treniranje Naive Bayes modela

Evaluacija performansi (accuracy, precision, recall, F1-score)

Interaktivna predikcija novih proizvoda
Uputstvo za pokretanje
1. Instalacija

Pre pokretanja, instalirajte neophodne biblioteke:

pip install pandas numpy scikit-learn scipy jupyter

2. Treniranje modela

Pokrenite train_model.py kako biste:

Učitavali i čistili podatke

Kreirali feature-e

Trenirali Naive Bayes model

Sačuvali model, TF-IDF vektorizator i LabelEncoder

python train_model.py

3. Predikcija nove kategorije

Pokrenite predict_category.py i unesite naziv proizvoda:

Primer interaktivnog unosa:

Enter product name: iphone 7 32gb gold
Predicted category: Mobile Phones

Evaluacija modela

Model je evaluiran na test skupu koristeći sledeće metrike:

Accuracy: ukupna preciznost predikcija

Classification report: precision, recall, F1-score po kategorijama

Confusion matrix: vizuelni pregled tačnosti predikcija

Naše testiranje pokazuje da Naive Bayes model postiže dobru preciznost za najčešće kategorije.

Biblioteke i tehnologije

Python 3.x

pandas, numpy

scikit-learn (TfidfVectorizer, LabelEncoder, MultinomialNB)

scipy (za sparse matricu)

Jupyter Notebook

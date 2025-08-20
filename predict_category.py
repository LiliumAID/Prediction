import pickle

# Učitavanje modela i transformera
model = pickle.load(open('product_category_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

def predict_category(product_title):
    X = vectorizer.transform([product_title])
    y_pred = model.predict(X)
    category = le.inverse_transform(y_pred)[0]
    return category

# Interaktivno testiranje
while True:
    product = input("Unesite naziv proizvoda (ili 'exit' za kraj): ")
    if product.lower() == 'exit':
        break
    category = predict_category(product)
    print(f"Predviđena kategorija: {category}")

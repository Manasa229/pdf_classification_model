import joblib
from sentence_transformers import SentenceTransformer

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier_model = joblib.load('models/pdf_classifier.joblib')

def classify_with_bert(pdf_text):

    message_embedding = transformer_model.encode(pdf_text)

    probabilities = classifier_model.predict_proba([message_embedding])[0]

    confidence = max(probabilities)

    predicted_class = classifier_model.predict([message_embedding])[0]

    return predicted_class, confidence


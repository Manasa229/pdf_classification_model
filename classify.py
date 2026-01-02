from datetime import datetime

from bert_processor import classify_with_bert
from llamaindex_processor import classify_document
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import pandas as pd

file_path = 'training/data_set/dataset.csv'
transformer = SentenceTransformer('all-MiniLM-L6-v2')
model_path='models/pdf_classifier.joblib'


async def classify_pdf(pdf_text, file_bytes,file_document_type):

    document_type,confidence = classify_with_bert(pdf_text)

    if confidence < 0.3:
        document_type,confidence = await classify_document(file_bytes,file_document_type)

        await add_to_training_data(pdf_text, document_type)

        df = pd.read_csv(file_path)
        if len(df) % 50 == 0:
            retrain_model()
    
    return document_type, confidence


async def add_to_training_data(text, document_type):
    """Add new document_typeed example to training dataset"""

    df = pd.read_csv(file_path)
    new_row = pd.DataFrame({'text': [text], 'document_type': [document_type]})
    df = pd.concat([df, new_row], ignore_index=True)

    # Save back to CSV
    df.to_csv(file_path, index=False)
    print(f"Added new training example: {document_type}")


def retrain_model():
    """Retrain the model with updated dataset"""

    df = pd.read_csv(file_path)

    embeddings = transformer.encode(df['text'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, df['document_type'], test_size=0.3, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    backup_path = f"{model_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        joblib.dump(joblib.load(model_path), backup_path)  # Backup old model
    except:
        pass

    joblib.dump(clf, model_path)
    print(f"Model retrained and saved to {model_path}")

    return clf
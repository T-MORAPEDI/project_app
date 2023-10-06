from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import os

import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords')

current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(current_dir, '..', 'ml_script', 'review_model.pkl')

model = joblib.load(model_filename)

vectorizer_filename = os.path.join(current_dir, '..', 'ml_script', 'review_vectorizer.pkl')

vectorizer = joblib.load(vectorizer_filename)

# Define a function for text preprocessing
def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()

    # Tokenize the text
    text = text.split()

    # Initialize a stemmer
    stemmer = SnowballStemmer("english")

    # Remove stopwords and apply stemming
    stuff_to_be_removed = set(stopwords.words('english')) | set(punctuation)
    text = [stemmer.stem(word) for word in text if word not in stuff_to_be_removed]

    # Join the processed tokens into a single string
    text = " ".join(text)

    return text


def predict_class(input_text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)

    # Transform the preprocessed text using the pre-fitted vectorizer
    input_vector = vectorizer.transform([preprocessed_text])

    # Make predictions
    predicted_class = model.predict(input_vector)[0]

    return predicted_class
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()

class Preprocessing:
    def __init__(self):
        pass

    def lemmatization(self, txt):
        if pd.isna(txt) or txt.strip() == "":
            return ""
        tokens = word_tokenize(str(txt))
        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
        lemmatized_words = " ".join(lemmatized_words)
        return lemmatized_words

    def vectorizer(self, df ):
        tfidvector = TfidfVectorizer(stop_words='english')
        matrix = tfidvector.fit_transform(df['lemmatized_text']).toarray().astype('float32')
        return matrix



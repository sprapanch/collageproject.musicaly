import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class Recommender:
    def __init__(self, df):
        self.df = df
        self.tf_idf_vector = TfidfVectorizer()

        # Fit TF-IDF once
        self.tfidf_matrix = self.tf_idf_vector.fit_transform(df['lemmatized_text']).toarray().astype('float32')

        # Build FAISS index
        self.index = self.build_faiss_index()

    def build_faiss_index(self):
        """Builds the FAISS index using precomputed TF-IDF vectors."""
        d = self.tfidf_matrix.shape[1]  # Number of features
        index = faiss.IndexFlatL2(d)
        index.add(self.tfidf_matrix)
        return index

    def recommend_by_song_name(self, song_name, top_k):
        """Find similar songs using FAISS and TF-IDF."""
        song_name = song_name.lower().strip()
        song_row = self.df[self.df['song'] == song_name]

        if song_row.empty:
            print(f"🛑 Song '{song_name}' not found in dataset. Using it as query.")
            query_vector = self.tf_idf_vector.transform([song_name]).toarray().astype('float32')
        else:
            query_vector = self.tf_idf_vector.transform([song_row['lemmatized_text'].values[0]]).toarray().astype(
                'float32')

        # Check if dimensions match
        if query_vector.shape[1] != self.index.d:
            print(f"⚠️ Dimension mismatch! Query: {query_vector.shape[1]}, Index: {self.index.d}")
            return []

        # Search FAISS
        _, indices = self.index.search(query_vector, top_k + 1)

        # Get recommended songs excluding the input
        recommended_songs = self.df.iloc[indices[0][1:]]['song'].tolist()
        print(recommended_songs)

        return recommended_songs

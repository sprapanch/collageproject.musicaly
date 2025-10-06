from flask import Flask, request, render_template
import pandas as pd
import faiss
import pickle
import requests
from Recommender import Recommender
from Preprocessing import Preprocessing

# Download required NLTK data
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')

app = Flask(__name__)

SPOTIFY_CLIENT_ID = "d92a0a6b0f2c409183580ee7d80b13f9"
SPOTIFY_CLIENT_SECRET = "4e75e43ebe6e41cf84b868d30b2f80cd"

# Function to get Spotify token
def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }
    response = requests.post(url, headers=headers, data=data)
    print(response)
    return response.json().get("access_token") if response.status_code == 200 else print("🛑 Failed to get Spotify token:", response.json())


# Function to get song details from Spotify
def get_spotify_details(song_name):
    token = get_spotify_token()
    if not token:
        return None

    search_url = f"https://api.spotify.com/v1/search?q={song_name}&type=track&limit=1"
    search_headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(search_url, headers=search_headers)

    if response.status_code != 200:
        return None

    tracks = response.json().get("tracks", {}).get("items", [])
    if tracks:
        track = tracks[0]
        return {
            "title": track["name"],
            "artist": track["artists"][0]["name"],
            "link": track["external_urls"]["spotify"],
            "image": track["album"]["images"][0]["url"],
        }
    return None


# Load Dataset
df = pd.read_csv("spotify_millsongdata.csv")
df = df.sample(5000).drop(columns=['link'], errors='ignore').reset_index(drop=True)

# Normalize song names
df['song'] = df['song'].str.lower().str.strip()

# Text Cleaning - Fill NaN values before preprocessing
df['text'] = df['text'].fillna('').str.lower().replace(r'\W+', ' ', regex=True)

preprocessor = Preprocessing()
df['lemmatized_text'] = df['text'].apply(preprocessor.lemmatization)

if df['lemmatized_text'].isnull().all() or df['lemmatized_text'].str.strip().eq("").all():
    raise ValueError("🛑 No valid text found for vectorization. Check preprocessing.")

# TF-IDF Vectorization
matrix = preprocessor.vectorizer(df)

# Ensure matrix is valid for FAISS
if matrix.shape[0] == 0 or matrix.shape[1] == 0:
    raise ValueError("🛑 FAISS index cannot be built with empty matrix.")

# Build FAISS index
#d = matrix.shape[1]  # Feature dimensions
#index = faiss.IndexFlatL2(d)  # L2 distance (use IndexFlatIP for cosine similarity)
#index.add(matrix)

# Save models
#faiss.write_index(index, "faiss_index.bin")
print("Faiss is built!")
#pickle.dump(df, open('df.pkl', 'wb'))
3
tf_idf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
#df = pickle.load(open("df.pkl", "rb"))
index = faiss.read_index("faiss_index.bin")
print("Now loading...")
recommender = Recommender(df=df)

#Example Usage
#input_song_name = input("Enter song name: ")
top_k = 5
#recommended_songs = recommender.recommend_by_song_name(input_song_name, top_k, index = index, df = df)

#print(f"🎵 Recommended songs similar to '{input_song_name}':\n {recommended_songs}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_song = request.form["song"]
        recommendations = recommender.recommend_by_song_name(input_song, top_k)
        song_data = []
        for song in recommendations:
            spotify_data = get_spotify_details(song)
            if spotify_data:
                song_data.append(spotify_data)

        print("DOne")

        if not recommendations:
            return render_template("index.html", recommendations=[], message="No similar songs found.")

        return render_template("index.html", recommendations=song_data)

    return render_template("index.html", recommendations=[])

if __name__ == "__main__":
    app.run(debug=False)

import streamlit as st
import pandas as pd
import requests
import pickle

# Load the processed data and similarity matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

# simple processing (you may already have this)
movies['overview'] = movies['overview'].fillna('')

cv = CountVectorizer(max_features=2000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

cosine_sim = cosine_similarity(vectors)

# Your TMDB API key
api_key = "525fb3cdbb9012071174dd036400b84b"

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    return movies[['title', 'movie_id']].iloc[movie_indices].assign(score=scores)

# Fetch movie poster from TMDB API
def fetch_poster(movie_id):
    try:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
        response = requests.get(url)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

# Streamlit UI
st.title("Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button('Recommend'):
    recommendations = get_recommendations(selected_movie)
    st.write("Top 10 recommended movies:")

    # Create a 2x5 grid layout
    for i in range(0, 10, 5):
        cols = st.columns(5)
        for col, j in zip(cols, range(i, i+5)):
            if j < len(recommendations):
                movie_title = recommendations.iloc[j]['title']
                movie_id = recommendations.iloc[j]['movie_id']
                score = recommendations.iloc[j]['score']
                poster_url = fetch_poster(movie_id)
                with col:
                    st.image(poster_url, width=130)
                    st.write(f"{movie_title} ({score:.2f})")
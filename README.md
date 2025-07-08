# Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on movie features like genres, keywords, cast, crew, and plot overview using the TMDB 5000 movie dataset.

## Features

- **Content-Based Filtering**: Recommends movies based on movie attributes (genres, cast, keywords, plot)
- **Text Processing**: Uses NLP techniques for feature extraction from movie descriptions
- **Cosine Similarity**: Calculates similarity between movies using vectorized features
- **Stemming**: Reduces words to root forms for better matching
- **Top-N Recommendations**: Returns top 5 similar movies for any given movie

## Dataset

Uses TMDB 5000 movie dataset containing:
- `tmdb_5000_movies.csv`: Movie metadata (title, overview, genres, keywords)
- `tmdb_5000_credits.csv`: Cast and crew information
- 4,806 movies after data cleaning

## Installation

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn nltk
pip install ast pickle
```

## Data Processing Pipeline

1. **Data Merging**: Combines movies and credits datasets
2. **Feature Selection**: Extracts relevant columns (title, overview, genres, keywords, cast, crew)
3. **JSON Parsing**: Converts string representations to lists using `ast.literal_eval()`
4. **Feature Engineering**: 
   - Extracts top 3 cast members
   - Extracts director from crew
   - Removes spaces from multi-word terms
5. **Text Processing**:
   - Combines all features into tags
   - Converts to lowercase
   - Applies Porter Stemming
6. **Vectorization**: Uses CountVectorizer with 5000 max features

## Quick Start

```python
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data
movies_df = pickle.load(open('movies.pkl', 'rb'))
similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies_df.iloc[i[0]].title)
    return recommended_movies

# Get recommendations
recommendations = recommend('Avatar')
print(recommendations)
```

## File Structure

```
movie-recommender/
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── models/
│   ├── movies.pkl
│   ├── movie_dict.pkl
│   └── similarity.pkl
├── Movie_Recommendation_System.ipynb
└── README.md
```

## Key Functions

### Data Preprocessing
```python
def convert(obj):
    """Convert JSON string to list of names"""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def fetch_director(obj):
    """Extract director from crew data"""
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def stem(text):
    """Apply Porter Stemming to text"""
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
```

### Recommendation Engine
```python
def recommend(movie):
    """Get top 5 similar movies"""
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

## Technical Implementation

- **Vectorization**: CountVectorizer with max_features=5000 and English stop words
- **Similarity Metric**: Cosine similarity for finding movie relationships
- **Feature Engineering**: Combines overview, genres, keywords, cast, and crew into single tag
- **Text Processing**: Lowercasing, stemming, and space removal for consistency

## Example Usage

```python
# Example recommendations
recommend('Avatar')        # Returns sci-fi/adventure movies
recommend('Batman Begins') # Returns superhero/action movies
```

## Model Performance

- **Dataset Size**: 4,806 movies after preprocessing
- **Feature Vector Size**: 5,000 dimensions
- **Similarity Matrix**: 4,806 x 4,806 cosine similarity matrix
- **Processing Time**: Fast inference due to precomputed similarity matrix

## Saved Models

- `movies.pkl`: Preprocessed movie DataFrame
- `movie_dict.pkl`: Movie data as dictionary
- `similarity.pkl`: Precomputed cosine similarity matrix

## Limitations

- Content-based approach only (no collaborative filtering)
- Requires exact movie title match
- Limited to movies in TMDB dataset
- No user preference learning
- Cold start problem for new movies

## Future Enhancements

- Add collaborative filtering
- Implement fuzzy string matching for movie titles
- Include user ratings and reviews
- Add movie popularity weighting
- Implement hybrid recommendation approach

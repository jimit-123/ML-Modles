# 🎬 CineMatch — AI Movie Recommendation System

CineMatch is an AI-powered movie recommendation system built using Machine Learning and content-based filtering techniques.

The project recommends movies based on genres, themes, cast, director style, language, ratings, and other movie metadata using TF-IDF vectorization and cosine similarity.

---

# 🚀 Features

* AI-powered movie recommendations
* Content-based filtering system
* TF-IDF feature extraction
* Cosine similarity recommendation engine
* Hyperparameter tuning using GridSearchCV
* Genre diversity enforcement
* Popularity bias mitigation
* Responsive cinematic UI
* Real-time movie search experience

---

# 🧠 Machine Learning Approach

The recommendation engine works by creating a weighted “feature soup” for every movie using:

* Genres
* Director
* Cast
* Keywords
* Overview
* Language
* Ratings
* Release year

These features are transformed using TF-IDF vectorization and compared using cosine similarity to find the most similar movies.

The model was evaluated using Genre Precision@5 metrics.

---

# 📊 Model Performance

| Metric                           | Score        |
| -------------------------------- | ------------ |
| Genre Precision@5                | 65%          |
| Hyperparameter Tuned Precision@5 | 72%          |
| Bias Gap                         | 0.118        |
| Dataset Size                     | 1152+ movies |

---

# 🛠 Tech Stack

## Machine Learning

* Python
* scikit-learn
* Pandas
* NumPy

## ML Concepts Used

* TF-IDF Vectorization
* Cosine Similarity
* Content-Based Filtering
* GridSearchCV
* MinMaxScaler
* Feature Engineering

## Frontend

* HTML
* CSS
* JavaScript

---

# 📂 Dataset

Dataset used:
MovieLens ML-100K

The dataset contains movie metadata, genres, ratings, and user interaction information.

---

# ⚙️ How It Works

1. User searches for a movie
2. The system converts movie metadata into weighted feature vectors
3. Cosine similarity calculates similarity scores
4. Top matching movies are recommended
5. Diversity and fairness adjustments are applied

---

# 🎯 Goals of This Project

This project was built to:

* Learn recommendation systems
* Explore NLP-based feature extraction
* Understand ML model evaluation
* Experiment with fairness in AI systems
* Combine frontend development with machine learning


---

# 🔗 Live Demo

https://zesty-klepon-18b468.netlify.app/

---

# 👨‍💻 Author

Jimit
Class 12 Student | AI & ML Enthusiast
Passionate about Artificial Intelligence, Machine Learning, and building real-world tech projects.

---

# 📌 Future Improvements

* Collaborative filtering integration
* Deep learning-based recommendations
* User accounts & watchlists
* Real TMDB API integration
* Personalized recommendation history
* Better multilingual recommendations

---

# ⭐ If you liked this project

Feel free to star the repository and connect with me on LinkedIn.

Linkdin - https://www.linkedin.com/in/jimit-makwana

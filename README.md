#  FlightRankPredictor

**Advanced Flight Recommendation System for Business Travelers**  
Kaggle Flight Rank 2025 · Ensemble Learning · Recommender Systems · Ranking Metrics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  Project Overview

FlightRankPredictor is a high-performance **machine learning pipeline** built for the [Flight Rank 2025 Kaggle competition](https://www.kaggle.com/competitions/flightrank-2025). It ranks flight options for users based on structured data and historical preferences, simulating a real-world travel recommendation engine.

The system integrates:

-  Advanced feature engineering
-  Intelligent ranking with `LightGBM`, `XGBoost`, `Logistic Regression`
-  Threshold tuning, SVD embeddings & group-wise normalization
-  Post-processing reranking based on business logic (e.g., penalizing duplicate flight numbers)
-  Ensemble methods for better generalization
-  Optuna hyperparameter tuning (for XGBoost ranking)

---

##  Features

-  **5M+ row scale support** using efficient memory management
-  **Optimized for NDCG@3** (Normalized Discounted Cumulative Gain)
-  Supports **ranking within user sessions** (`ranker_id`-grouped permutations)
-  Includes **re-ranking heuristics** to handle duplicate itineraries
-  Hyperparameter tuning with **Optuna**
-  Feature types: Categorical, Duration Parsing, Count-Based, Price Bins, Embeddings

---

##  Model Architecture

| Model                | Type              | Framework    | Notes                           |
|---------------------|-------------------|--------------|---------------------------------|
| LightGBM            | LambdaRank        | LightGBM     | Optimized for NDCG@3            |
| XGBoost             | Pairwise Ranking  | XGBoost      | Tuned via Optuna                |
| Logistic Regression | Binary Classifier | Scikit-learn | Blended as fallback signal      |
| Post-Processing     | Rule-Based Ranker | Polars       | Penalizes duplicate flight hash |
| Embeddings          | SVD Matrix Factor | Scikit-learn | For carrier-user interaction    |

---



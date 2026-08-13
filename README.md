# E-Commerce Recommendation Engine

An end-to-end machine learning recommendation system built using real-world e-commerce interaction data.

## Objective

Build a personalized recommendation engine that recommends products based on user interaction behavior.

## Dataset

The project uses the publicly available RetailRocket e-commerce recommendation dataset.

## Current Approach

- Data preprocessing
- Popularity-based baseline
- Item-item collaborative filtering
- Hybrid recommendation
- Hyperparameter experimentation
- Precision@K
- Recall@K
- NDCG@K
- MLflow experiment tracking

## Current Best Result

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Hybrid | 0.00704 | 0.03953 | 0.02890 |

Evaluation was performed on 10,000 users.

## Tech Stack

Python, Pandas, NumPy, SciPy, Scikit-learn, MLflow

## Project Status

Currently under development.
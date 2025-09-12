# 🍽️ Food Recommendation Suite — Supervised ML Projects

A collection of six supervised machine learning projects focused on food ordering and recommendation systems. These notebooks use sample data and are configured to run both locally and in Google Colab.

## Overview

This repository contains end-to-end implementations of common recommendation and marketing tasks, including reorder prediction, rating estimation, personalized ranking, co-purchase detection, bundle value estimation, and hybrid recommendation approaches.

## Project Structure

```
food-recommendation-suite/
│
├── notebooks/              
│   ├── project1_next_order_reorder.ipynb
│   ├── project2_rating_regression.ipynb
│   ├── project3_supervised_topn_ranking.ipynb
│   ├── project4_copurchase_prediction.ipynb
│   ├── project5_deal_value_regression.ipynb
│   └── project6_popularity_personalized_blend.ipynb
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Projects

1. **Next-Order Reorder Prediction** - Uses RandomForestClassifier to predict customer retention signals
2. **Rating Regression** - Implements GradientBoostingRegressor to predict user-item ratings
3. **Supervised Top-N Ranking** - Creates personalized recommendation lists using GradientBoostingRegressor
4. **Co-purchase Prediction** - Identifies complementary items with RandomForestClassifier
5. **Deal Value Regression** - Estimates bundle value using GradientBoostingRegressor
6. **Popularity + Personalized Blend** - Combines global and personalized signals with StackingClassifier

## Quick Start

### Local Setup
```bash
git clone https://github.com/sinacipher/food-recommendation-suite.git
cd food-recommendation-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Google Colab
Open any notebook directly in Colab using this pattern:
```
https://colab.research.google.com/github/sinacipher/food-recommendation-suite/blob/main/notebooks/project1_next_order_reorder.ipynb
```

## Dependencies

The core requirements include:
```
pandas
numpy
scikit-learn
matplotlib
jupyter

```

## Usage Examples

# Predict reorder probability
sample = {
    'cuisine': 'Pizza',
    'price': 12.5,
    'discount': 10,
    'delivery_time': 25,
    'previous_orders': 3,
    'days_since_last': 10,
    'order_hour': 19
}
# predict_reorder(sample) returns: {'prediction': 1, 'probability': 0.72}

## Demo Guide

For a quick 5-minute demo:
1. Run Project 1 to show reorder prediction with ROC curves
2. Demonstrate Project 3's personalized ranking with sample user output
3. Show bundle recommendations from Project 5
4. Optionally showcase a simple API or UI demo for live testing

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or suggestions, please reach out to [sina.cipher11228@gmail.com](mailto:sina.cipher11228@gmail.com).

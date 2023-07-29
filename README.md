# Price Recommendation Engine for Online Sellers using LightGBM
![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/01f1efda01281a9f15e19c82590fbc32c3db37c4/head1.gif?raw=true)

## Table of Contents
1. [Introduction](#Introduction)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

<a name="Introduction"/>
## 1. Introduction

This is a machine learning project that aims to build a price recommendation engine for online sellers using the LightGBM algorithm. The objective is to predict the appropriate selling price for different products listed by sellers on an e-commerce platform.

## 2. Project Overview

In today's e-commerce world, setting the right price for a product is crucial for sellers to attract buyers and maximize their profits. However, determining the optimal price can be challenging due to various factors like product category, brand, condition, shipping cost, and more. This project leverages machine learning to help sellers make data-driven decisions on pricing.

## 3. Dataset

The dataset used in this project is obtained from the "Mercari Price Suggestion Challenge" on Kaggle. It contains product listings with features like `name`, `item_condition_id`, `category_name`, `brand_name`, `shipping`, `item_description`, and the target variable `price`. The data is split into training and testing sets.
* ID: the id of the listing
* Name: the title of the listing
* Item Condition: the condition of the items provided by the seller
* Category Name: category of the listing
* Brand Name: brand of the listing
* Shipping: whether or not shipping cost was provided
* Item Description: the full description of the item
* Price: the price that the item was sold for. This is the target variable that you will predict. The unit is USD.

Dataset-source: [Kaggle - Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data)

## 4. Data Preprocessing

Before building the model, the data undergoes preprocessing steps to handle missing values and clean the text data. The preprocessing includes:
- Handling missing values in `category_name`, `brand_name`, and `item_description` fields.
- Converting `item_condition_id`, `category_name`, and `brand_name` to categorical variables.

## 5. Feature Engineering

Feature engineering is a crucial step to create meaningful predictors for the model. Key features are created from the existing data, such as:
- Count vectorization of `name` and `category_name`.
- TF-IDF vectorization of `item_description`.
- One-hot encoding of `item_condition_id` and `shipping`.

## 6. Model Training

LightGBM, a gradient boosting framework, is used to build the price recommendation model. It is chosen for its efficiency and ability to handle large datasets. The model is trained on the preprocessed data and tuned using hyperparameters for optimal performance.

## 7. Evaluation

The model's performance is evaluated using the Root Mean Squared Error (RMSE) metric on the test set. Lower RMSE indicates better prediction accuracy. Additionally, a visualization of the price distribution is provided.

## 8. Installation

To run the project, follow these steps:
1. Clone the repository: `git clone https://github.com/yourusername/price-recommendation.git`
2. Navigate to the project directory: `cd price-recommendation`
3. Install the required dependencies: `pip install -r requirements.txt`

## 9. Usage

To use the price recommendation engine, follow these steps:
1. Prepare your product data in a CSV format similar to the training dataset.
2. Load the data and preprocess it using the `heandel_missing_inplace()` and `cutting()` functions.
3. Transform the text data using vectorization methods like CountVectorizer and TF-IDF.
4. Load the trained LightGBM model and use it to predict prices for your products.
5. Analyze the recommendations and make informed pricing decisions.

## 10. Contributing

Contributions to this project are welcome. If you find any issues or want to add new features, please submit a pull request. For major changes, open an issue first to discuss the proposed changes.

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

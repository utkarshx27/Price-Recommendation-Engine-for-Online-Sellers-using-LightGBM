# Price Recommendation Engine for Online Sellers using LightGBM
<p align="center">
  <img src="https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/01f1efda01281a9f15e19c82590fbc32c3db37c4/head1.gif?raw=true" alt="Price Recommendation Engine">
</p>

## Table of Contents
1. [Introduction](#Introduction)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Hyperparameter Tuning](hyperparameter-tuning)
8. [Evaluation](#evaluation)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Contributing](#contributing)
12. [License](#license)

<a name="Introduction"/>

## 1. Introduction

This is a machine learning project that aims to build a price recommendation engine for online sellers using the LightGBM algorithm. The objective is to predict the appropriate selling price for different products listed by sellers on an e-commerce platform.

<a name="Project-Overview"/>

## 2. Project Overview

In today's e-commerce world, setting the right price for a product is crucial for sellers to attract buyers and maximize their profits. However, determining the optimal price can be challenging due to various factors like product category, brand, condition, shipping cost, and more. This project leverages machine learning to help sellers make data-driven decisions on pricing.

<a name="Dataset"/>

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

<a name="Data-Preprocessing"/>

## 4. Data Preprocessing

Before building the model, the data undergoes preprocessing steps to handle missing values and clean the text data. The preprocessing includes:
- Handling missing values in `category_name`, `brand_name`, and `item_description` fields.
- Converting `item_condition_id`, `category_name`, and `brand_name` to categorical variables.

<a name="Feature-Engineering"/>

## 5. Feature Engineering

Feature engineering is a crucial step to create meaningful predictors for the model. Key features are created from the existing data, such as:
- Count vectorization of `name` and `category_name`.
- TF-IDF vectorization of `item_description`.
- One-hot encoding of `item_condition_id` and `shipping`.

<a name="Model-Training"/>

## 6. Model Training

LightGBM, a gradient boosting framework, is used to build the price recommendation model. It is chosen for its efficiency and ability to handle large datasets. The model is trained on the preprocessed data and tuned using hyperparameters for optimal performance.

<a name="Hyperparameter-Tuning"/>

## 7. Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing the model's performance. In this case, the hyperparameters to be tuned are 'max_depth' and 'num_leaves'. Grid search with cross-validation (5-fold) is employed to find the best combination of these hyperparameters.
```
param_grid = {
    'max_depth': [3, 5, 7],
    'num_leaves': [50, 100, 150],
}

grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

best_params = grid_search.best_params_
gbm = lgb.LGBMRegressor(application='regression', metric='RMSE', verbosity=-1, **best_params)
gbm.fit(X, y)

```
After the hyperparameter tuning process, the best set of hyperparameters is obtained and used to reinitialize the LightGBMRegressor model.

<a name="Evaluation"/>

## 8. Evaluation

The model's performance is assessed using several metrics on the test set:
```
y_pred = gbm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

```
The metrics used for evaluation are:

- Mean Squared Error (MSE): The mean squared difference between the actual and predicted target values. Lower MSE indicates better performance.
- Mean Absolute Error (MAE): The mean absolute difference between the actual and predicted target values. Lower MAE indicates better performance.
- R^2 Score: The coefficient of determination, which measures the proportion of variance in the target variable explained by the model. A value closer to 1 indicates better model fit.

Additionally, to visualize the model's performance and assess potential overfitting, a learning curve is plotted:
```
train_sizes, train_scores, test_scores = learning_curve(
    gbm, X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Error')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation Error')
plt.xlabel('Training Size')
plt.ylabel('Negative Mean Squared Error')
plt.legend(loc='best')
plt.title('Learning Curve')
plt.show()

```
The learning curve helps to analyze how the model's performance changes as the training data size increases and can give insights into potential overfitting or underfitting.

<a name="Installation"/>

## 9. Installation

To run the project, follow these steps:
1. Clone the repository: `git clone https://github.com/yourusername/price-recommendation.git`
2. Navigate to the project directory: `cd price-recommendation`
3. Install the required dependencies: `pip install -r requirements.txt`

<a name="Usage"/>

## 10. Usage

To use the price recommendation engine, follow these steps:
1. Prepare your product data in a CSV format similar to the training dataset.
2. Load the data and preprocess it using the `heandel_missing_inplace()` and `cutting()` functions.
3. Transform the text data using vectorization methods like CountVectorizer and TF-IDF.
4. Load the trained LightGBM model and use it to predict prices for your products.
5. Analyze the recommendations and make informed pricing decisions.

<a name="Contributing"/>

## 11. Contributing

Contributions to this project are welcome. If you find any issues or want to add new features, please submit a pull request. For major changes, open an issue first to discuss the proposed changes.

<a name="License"/>

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Price-Recommendation-for-Online-Sellers

E-commerce platforms today are extensively driven by machine learning algorithms, right from quality checking and inventory management to sales demographics and product recommendations, all use machine learning. One more interesting business use case that e-commerce apps and websites are trying to solve is to eliminate human interference in providing price suggestions to the sellers on their marketplace to speed up the efficiency of the shopping website or app. That's when price recommendation using machine learning comes to play.

## Dataset Features

* ID: the id of the listing
* Name: the title of the listing
* Item Condition: the condition of the items provided by the seller
* Category Name: category of the listing
* Brand Name: brand of the listing
* Shipping: whether or not shipping cost was provided
* Item Description: the full description of the item
* Price: the price that the item was sold for. This is the target variable that you will predict. The unit is USD.

Dataset-source: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data

![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/01f1efda01281a9f15e19c82590fbc32c3db37c4/head1.gif?raw=true)
## Import Packages

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
```
```
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
```
## Visulization
### Price Distribution Chart
```
plt.subplot(1, 2, 1)
(train['price']).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'black', range = [0, 250],color = 'pink')
plt.xlabel('price', fontsize=12)
plt.title('Price Distribution', fontsize=12)
plt.subplot(1, 2, 2)
np.log(train['price']+1).plot.hist(bins=50, figsize=(12,6), edgecolor='black')
plt.xlabel('log(price+1)', fontsize=12)
plt.title('Price Distribution', fontsize=12)
```
![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/52a7e874abf2e0d19f339dda9a5b84d9295cc714/price_img_dist.png?raw=true)


### Item Condition Representation
```
sns.boxplot(x = 'item_condition_id', y =np.log(train['price']+1), data = train, palette = sns.color_palette('RdBu',5))
```
![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/89205c47be4c5a09ca383477f04765b6b56cca4c/chart2.png?raw=true)

### Top 20 Brand Distribution Representation
```
b20 = train['brand_name'].value_counts()[0:20].reset_index().rename(columns={'index': 'brand_name', 'brand_name':'count'})
ax = sns.barplot(x="brand_name", y="count", data=b20)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Top 20 Brand Distribution', fontsize=15)
plt.show()
```
![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/dd22de77b1a0e17bbeffdf4a05dcce2df5e58d25/chart3.png?raw=true)


## Data pre-processing (helper functions)

```
def heandel_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].replace('No description yet,''missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)
    
    
def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]

    
def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
```
## Transforming Data
```
#Count vectorize name and category name columns.
cv = CountVectorizer(min_df=NAME_MIN_DF)   
x_name = cv.fit_transform(merge['name'])     
cv = CountVectorizer()
x_category = cv.fit_transform(merge['category_name'])
```
```
#TF-IDF Vectorize item_description column.
tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 3), stop_words='english')
X_description = tv.fit_transform(merge['item_description'])
```
```
#Label binarize brand_name column.
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
```
## Parameters Used
```
params = {
    'learning_rate': 0.75,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 100,
    'verbosity': -1,
    'metric': 'RMSE',
}
```
* Use ‘regression’ as application as we are dealing with a regression problem.
* Use ‘RMSE’ as metric because this is a regression problem.
* num_leaves=100 as our data is relative big.
* Use “max_depth to avoid overfitting.
* Use “verbosity to control the level of LightGBM’s verbosity.
* learning_rate determines the impact of each tree on the final outcome.
## Train

```
gbm = lgb.train(params, train_set=train_X, num_boost_round=3200,verbose_eval=100)
```
## Test
```
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
```
## Evaluation
```
mean_squared_error(y_test, y_pred)
```
0.21321232298146012

# Price-Recommendation-for-Online-Sellers
## Description- 
E-commerce platforms today are extensively driven by machine learning algorithms, right from quality checking and inventory management to sales demographics and product recommendations, all use machine learning. One more interesting business use case that e-commerce apps and websites are trying to solve is to eliminate human interference in providing price suggestions to the sellers on their marketplace to speed up the efficiency of the shopping website or app. That's when price recommendation using machine learning comes to play.

![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/01f1efda01281a9f15e19c82590fbc32c3db37c4/head1.gif?raw=true)
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
plt.subplot(1,2,1)
(train['price']).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white', range= [0,200])
plt.xlabel('Price', fontsize=12)
plt.title('Price Distribution', fontsize=12)
plt.subplot(1,2,2)
np.log(train['price']+1).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white')
plt.xlabel('log(Price)+1', fontsize=12)
plt.title('Price Distribution', fontsize=12)
```
![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/89205c47be4c5a09ca383477f04765b6b56cca4c/chart1.png?raw=true)


### Item Condition Representation
```
sns.boxplot(x = 'item_condition_id', y =np.log(train['price']+1), data = train, palette = sns.color_palette('RdBu',5))
```
![alt text](https://github.com/utkarshh27/Price-Recommendation-for-Online-Sellers/blob/89205c47be4c5a09ca383477f04765b6b56cca4c/chart2.png?raw=true)



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

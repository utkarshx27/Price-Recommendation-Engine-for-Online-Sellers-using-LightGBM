#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import gc
import time




# In[3]:


df = pd.read_csv(r"C:\Users\UTKARSH SINGH\Documents\Projects\Price Recommendation for Online Sellers using LightGBM\mercari-price-suggestion-challenge\train.tsv", sep = '\t')


# In[4]:


df.head(10)


# In[5]:


#df.isnull()


# In[6]:


train, test = train_test_split(df, test_size=0.2, random_state=42)    #30 krna h
train.shape, test.shape


# In[7]:


plt.subplot(1,2,1)
(train['price']).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white', range= [0,200])
plt.xlabel('Price', fontsize=12)
plt.title('Price Distribution', fontsize=12)
plt.subplot(1,2,2)
np.log(train['price']+1).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white')
plt.xlabel('log(Price)+1', fontsize=12)
plt.title('Price Distribution', fontsize=12)


# In[8]:


train['shipping'].value_counts()/len(train) *100


# In[9]:


fee_buyer = train.loc[df['shipping'] ==0, 'price']
fee_seller = train.loc[df['shipping']==1, 'price']
print("The avg. Price is {}".format(round(fee_seller.mean(), 2)), "when seller pays the shipping");
print("The avg. Price is {}".format(round(fee_buyer.mean(), 2)), "when buyer pays the shipping")


# In[ ]:





# In[10]:


train['item_condition_id'].unique()


# In[11]:


sns.boxplot(x = 'item_condition_id', y =np.log(train['price']+1), data = train, palette = sns.color_palette('RdBu',5))


# In[12]:


b20 = train['brand_name'].value_counts()[0:20].reset_index().rename(columns={'index': 'brand_name', 'brand_name':'count'})
ax = sns.barplot(x="brand_name", y="count", data=b20)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_title('Top 20 Brand Distribution', fontsize=15)
plt.show()


# In[39]:


NUM_BRANDS = 4000                  #remember
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 500000


# In[43]:


train['category_name'].isnull().sum()


# In[44]:


train['brand_name'].isnull().sum()


# In[45]:


train['item_description'].isnull().sum()


# In[46]:


train['price'].isnull().sum()


# In[47]:


train['item_condition_id'].isnull().sum()


# In[48]:


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


# In[49]:


df = pd.read_csv(r"C:\Users\UTKARSH SINGH\Documents\Projects\Price Recommendation for Online Sellers using LightGBM\mercari-price-suggestion-challenge\train.tsv", sep = '\t')
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
test_new = test.drop('price', axis=1)
y_test = np.log1p(test["price"])
train = train[train.price !=0].reset_index(drop=True)


# In[52]:


nrow_train = train.shape[0]
y = np.log1p(train["price"])
merge: pd.DataFrame = pd.concat([train, test_new])


# In[53]:


heandel_missing_inplace(merge)
cutting(merge)
to_categorical(merge)


# In[63]:


cv = CountVectorizer(min_df=NAME_MIN_DF)
x_name = cv.fit_transform(merge['name'])
cv = CountVectorizer()
x_category = cv.fit_transform(merge['category_name'])


# In[66]:


tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 3), stop_words='english')
X_description = tv.fit_transform(merge['item_description'])


# In[70]:


lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])


# In[73]:


X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)


# In[75]:


sparse_merge = hstack((X_dummies, X_description, X_brand, x_category, x_name)).tocsr()


# In[76]:


mask = np.array(np.clip(sparse_merge.getnnz(axis=0) -1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]


# In[77]:


X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]


# In[78]:


train_X =lgb.Dataset(X, label=y)


# In[80]:


params = {
    'learning_rate': 0.75,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 100,
    'verbosity': -1,
    'metric': 'RMSE',
}


# In[81]:


gbm = lgb.train(params, train_set=train_X, num_boost_round=3200,verbose_eval=100)


# In[ ]:


y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)


# In[83]:


mean_squared_error(y_test, y_pred)


# In[5]:


# Score was 0.49 something


# In[20]:


#!jupyter nbconvert --to pdf price_recommendation.ipynb


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





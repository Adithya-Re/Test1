#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd


# In[35]:


df=pd.read_csv(r'C:\Users\bhask\OneDrive\Documents\KN-ML\Ineuron Materials\train.csv')
df2=pd.read_csv(r'C:\Users\bhask\OneDrive\Documents\KN-ML\Ineuron Materials\train.csv')


# In[36]:


df.sample(1)


# In[37]:


df=df.drop(['ID'],axis=1)


# In[38]:


df.isnull().sum()


# In[39]:


def impute_nan(df,i):
    df[i+"_random"]=df[i]
    random_sample=df[i].dropna().sample(df[i].isnull().sum(),random_state=0)
    random_sample.index=df[df[i].isnull()].index
    df.loc[df[i].isnull(),i+'_random']=random_sample
    df.drop([i],axis=1,inplace=True)


# In[40]:


impute_nan(df,'Credit_Product')


# In[41]:


df.isnull().sum()


# In[42]:


b=list(df2['Region_Code'].unique())
print(len(b))


# In[43]:


df_age=df[['Gender','Credit_Product_random','Is_Active']]


# In[44]:


df_age = pd.get_dummies(df_age,drop_first=True)


# In[45]:



df.drop(['Gender'],axis=1,inplace=True)
df.drop(['Credit_Product_random'],axis=1,inplace=True)
df.drop(['Is_Active'],axis=1,inplace=True)


# In[46]:



df=pd.concat([df,df_age],axis=1)


# In[47]:


df.sample(1)


# In[48]:


def ordinal(df,i,j):
    ordinal_labels=df.groupby([i])[j].mean().sort_values().index
    enumerate(ordinal_labels,0)
    ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
    ordinal_labels2
    df[i+'_new']=df[i].map(ordinal_labels2)
    df.drop(i,inplace=True,axis=1)


# In[49]:


ordinal(df,'Region_Code','Is_Lead')


# In[50]:


df.sample(1)


# In[51]:


ordinal(df,'Occupation','Is_Lead')


# In[52]:


df.sample(1)


# In[53]:


df.groupby(['Channel_Code'])['Is_Lead'].mean()


# In[54]:


ordinal(df,'Channel_Code','Is_Lead')


# In[55]:


df.sample(2)


# In[56]:


df.shape


# In[57]:


x=df.drop(labels='Is_Lead', axis=1)
y= df['Is_Lead']


# In[58]:


x.head()


# In[59]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_data=scaler.fit_transform(x)


# In[60]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(scaled_data,y,test_size=0.3,random_state=42)


# In[61]:


# fit model to training data
from xgboost import XGBClassifier
model = XGBClassifier(objective='binary:logistic')
model.fit(train_x, train_y,verbose=True,early_stopping_rounds=10,eval_metric='aucpr',eval_set=[(test_x,test_y)])


# In[62]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(train_x)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(train_y,predictions)
accuracy


# In[63]:


y_pred = model.predict(test_x)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(test_y,predictions)
accuracy


# In[64]:


from sklearn.model_selection import GridSearchCV


# In[65]:


param_grid={
   
    'learning_rate':[0.05,0.1,0.15],
    'max_depth': [4,5,6],
    'n_estimators':[70,85,100]
    
}


# In[66]:


grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid, verbose=3)


# In[67]:


grid.fit(train_x,train_y,verbose=True,early_stopping_rounds=10,eval_metric='aucpr',eval_set=[(test_x,test_y)])


# In[71]:


grid.best_params_


# In[70]:


new_model=XGBClassifier(learning_rate= 0.1, max_depth= 6, n_estimators= 85)
new_model.fit(train_x, train_y,verbose=True,early_stopping_rounds=10,eval_metric='aucpr',eval_set=[(test_x,test_y)])


# In[72]:


from sklearn.metrics import accuracy_score
y_pred = new_model.predict(train_x)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(train_y,predictions)
accuracy


# In[73]:


y_pred = new_model.predict(test_x)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(test_y,predictions)
accuracy


# In[74]:


import pickle
filename = 'xgboost_model2.pickle'
pickle.dump(new_model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))


# In[164]:


dft=pd.read_csv(r'C:\Users\bhask\OneDrive\Documents\KN-ML\Ineuron Materials\test.csv')
df=pd.read_csv(r'C:\Users\bhask\OneDrive\Documents\KN-ML\Ineuron Materials\train.csv')


# In[165]:


dft.sample(1)


# In[166]:


dft_id=dft['ID']


# In[169]:


dft.isnull().sum()


# In[170]:


def impute_nan(df,i):
    df[i+"_random"]=df[i]
    random_sample=df[i].dropna().sample(df[i].isnull().sum(),random_state=0)
    random_sample.index=df[df[i].isnull()].index
    df.loc[df[i].isnull(),i+'_random']=random_sample
    df.drop([i],axis=1,inplace=True)


# In[171]:


impute_nan(dft,'Credit_Product')


# In[172]:


dft.isnull().sum()


# In[173]:


dft_age=dft[['Gender','Credit_Product_random','Is_Active']]


# In[174]:


dft_age = pd.get_dummies(dft_age,drop_first=True)


# In[175]:


dft.drop(['Gender'],axis=1,inplace=True)
dft.drop(['Credit_Product_random'],axis=1,inplace=True)
dft.drop(['Is_Active'],axis=1,inplace=True)


# In[176]:


dft=pd.concat([dft,dft_age],axis=1)


# In[177]:


dft.size


# In[178]:


def ordinal1(df,i,j):
    ordinal_labels=df.groupby([i])[j].mean().sort_values().index
    enumerate(ordinal_labels,0)
    return ordinal_labels


# In[179]:


def ordinal2(df,i,j):
    
    ordinal_labels2={k:i for i,k in enumerate(j,0)}
    ordinal_labels2
    df[i+'_new']=df[i].map(ordinal_labels2)
    df.drop(i,inplace=True,axis=1)


# In[180]:


j=ordinal1(df,'Region_Code','Is_Lead')


# In[181]:


ordinal2(dft,'Region_Code',j)


# In[182]:


j=ordinal1(df,'Occupation','Is_Lead')


# In[183]:


ordinal2(dft,'Occupation',j)


# In[184]:


j=ordinal1(df,'Channel_Code','Is_Lead')


# In[185]:


ordinal2(dft,'Channel_Code',j)


# In[186]:


dft=dft.drop('ID',axis=1)


# In[187]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_data=scaler.fit_transform(dft)


# In[188]:


loaded_model = pickle.load(open('xgboost_model2.pickle', 'rb'))


# In[189]:


pred=loaded_model.predict(scaled_data)


# In[190]:


df_pred = pd.DataFrame(pred, columns = ['Is_Lead'])


# In[191]:


df_pred.head()


# In[192]:


df5=pd.concat([dft_id,df_pred],ignore_index=True,axis=1)


# In[193]:


df5.to_csv(r'C:\Users\bhask\OneDrive\Documents\KN-ML\Ineuron Materials\pred_b.csv', index = False)


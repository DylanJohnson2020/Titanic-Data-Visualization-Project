#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series, DataFrame


# In[2]:


titanic_df = pd.read_csv('train.csv')
titanic_df.head()


# In[3]:


titanic_df.info()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


sns.catplot(x='Sex',data=titanic_df,kind='count')


# In[12]:


sns.catplot(x='Sex',data=titanic_df,kind='count',hue='Pclass')


# In[13]:


sns.catplot(x='Pclass',data=titanic_df,kind='count',hue='Sex')


# In[14]:


def male_female_child(passenger):
    age,sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[16]:


titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[17]:


titanic_df[0:10]


# In[18]:


sns.catplot(x='Pclass',data=titanic_df,hue='person',kind='count')


# In[25]:


titanic_df['Age'].hist(bins=70,edgecolor='black')


# In[26]:


titanic_df['Age'].mean()


# In[27]:


titanic_df['person'].value_counts()


# In[28]:


fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[29]:


fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[30]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[31]:


titanic_df.head()


# In[32]:


deck = titanic_df['Cabin'].dropna()


# In[33]:


deck.head()


# In[38]:


levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']

cabin_df['Cabin'].values.sort()

sns.catplot(x='Cabin',data=cabin_df,palette='winter_d',kind='count')
    


# In[39]:


cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot(x='Cabin',data=cabin_df,palette='summer',kind='count')
    


# In[40]:


titanic_df.head()


# In[43]:


sns.catplot(x='Embarked',data=titanic_df,hue='Pclass',kind='count',order=['C','Q','S'])


# In[44]:


titanic_df.head()


# In[46]:


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[48]:


titanic_df['Alone']


# In[49]:


titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone']==0] = 'Alone'


# In[50]:


titanic_df.head()


# In[51]:


sns.catplot(x='Alone',data=titanic_df,palette='Blues',kind='count')


# In[55]:


titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.catplot(x='Survivor',data=titanic_df,palette='Set1',kind='count')


# In[63]:


sns.catplot(x='Pclass',y='Survived',data=titanic_df,hue='person',kind='point')


# In[66]:


sns.lmplot(x='Age',y='Survived',data=titanic_df,)


# In[67]:


sns.lmplot(x='Age',y='Survived',data=titanic_df,hue='Pclass',palette='winter')


# In[68]:


generations = [10,20,40,60,80]

sns.lmplot(x='Age',y='Survived',data=titanic_df,hue='Pclass',palette='winter',x_bins=generations)


# In[69]:


sns.lmplot(x='Age',y='Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)


# In[70]:


titanic_df.head()


# In[73]:


cabin_df


# In[100]:


levels = pd.Series(levels)
titanic_df['Levels'] = levels
titanic_df.head()


# In[101]:


sns.catplot(x='Survived',hue='Levels',data=titanic_df,palette='winter', kind='count')


# In[102]:


sns.catplot(x='Survived',hue='Alone',data=titanic_df,palette='winter',kind='count')


# In[ ]:





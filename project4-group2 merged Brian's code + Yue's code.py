#!/usr/bin/env python
# coding: utf-8

# # Project 2: Group 4
# ## Exploratory Data Analysis (EDA)

# In[2]:


# Imports
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[3]:


# Load Data
df = pd.read_csv("cardata.csv")
# To display the top 5 rows 
df.head(5)   


# In[4]:


# List the different types of data
df.dtypes


# ### Goals for EDA
# We'd like to understand how the make up and branding of a car correlates with popularity \
# Irrelevant columns: Engine Cylinders, Engine Fuel Type, Engine HP, Transmission Type 

# In[6]:


# Drop irrelevant columns
df = df.drop(['Engine Fuel Type', 'Engine Cylinders', 'Engine HP', 'Transmission Type'], axis=1)
df.head(5)


# In[7]:


# Rename columns
df = df.rename(columns={"Driven_Wheels": "Drive Type","highway MPG": "MPG:H", "city mpg": "MPG:C", "MSRP": "List Price" })
df.head(5)


# In[8]:


# List number of duplicate rows
print("Before: number of rows: ", df.shape)

# Drop the duplicates
df = df.drop_duplicates()
print("After: number of rows: ", df.shape)

# Drop Missing/Null Values
print("Sum of null values: ", df.isnull().sum())
df = df.dropna()    # Dropping the missing values.
df.count()



# In[9]:


display("Head", df.head(5))
display("Tail", df.tail(5))


# In[10]:


# Detect and remove outliers
# Let's take a look at MPG:H, MPG:C, Populatiry, and List Price
sns.boxplot(x=df['MPG:H'])


# In[11]:


sns.boxplot(x=df['MPG:C'])


# In[12]:


sns.boxplot(x=df['Popularity'])


# In[13]:


sns.boxplot(x=df['List Price'])


# In[14]:


# Now let's remove outliers from these columns
# By setting numeric_only=True, we're only looking at the columns with numeric data and removing those outliers

Q1 = df.quantile(0.25, numeric_only=True)
Q3 = df.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1
print(IQR)


# In[15]:


# Now let's remove the outliers

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=[float, int])

# Reindex Q1, Q3, and IQR to match the columns in numeric_df
Q1 = Q1.reindex(numeric_df.columns)
Q3 = Q3.reindex(numeric_df.columns)
IQR = IQR.reindex(numeric_df.columns)

# Create the mask using the reindexed values and align it with df's index
mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
mask = mask.reindex(df.index, fill_value=False)

# Apply the mask to the original df
df = df[mask]
df.shape


# In[16]:


import plotly.graph_objs as go
import plotly.express as px
 
# Most popular make of car bar chart
sns.countplot(data=df,x="Make",palette="CMRmap", order=df["Make"].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel("Make",fontsize=10,color="black")
plt.ylabel("Number of Cars",fontsize=10,color="black")
plt.title("Car Make",color="black")
plt.show()


# In[17]:


# Most popular Vehicle Style
sns.countplot(data=df,x="Vehicle Style",palette="CMRmap", order=df["Vehicle Style"].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel("Vehicle Style",fontsize=10,color="black")
plt.ylabel("Number of Cars",fontsize=10,color="black")
plt.title("Car Style",color="black")
plt.show()


# In[18]:


#Top 20 Number of Cars by Make
df.Make.value_counts().nlargest(20).plot(kind='pie',figsize=(10,5))
plt.title("Top 20 Number of Cars by Make")
plt.ylabel("Nummber of Cars")
plt.xlabel("Make")


# In[36]:


#Top 10 Number of Card by Make
df.Make.value_counts().nlargest(10).plot(kind='barh',figsize=(20,10))
plt.title("Top 10 Number of Cars by Make")
plt.ylabel("Nummber of Cars")
plt.xlabel("Make")


# In[ ]:





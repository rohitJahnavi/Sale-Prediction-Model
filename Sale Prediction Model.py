#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('ecommerce_product_dataset.csv')


# In[3]:


df


# # Display basic information about the dataset

# In[4]:


df.info()


# In[5]:


df.head()


# # Data Preprocessing
# * Handle Missing values

# In[6]:


df.fillna(method="ffill", inplace= True)


# In[7]:


print("Missing values handled!")


# * Encode categorical variables if any

# In[8]:


if df.select_dtypes(include = "object").shape[1]>0:
    df=pd.get_dummies(df, drop_first= True)


# * Feature and target selection

# In[9]:


x = df.drop(columns=['Sales'])
y = df['Sales']


# * Train Test Split

# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[11]:


x_train, x_test, y_train, y_test


# # Model Traning
# * Initialize and train Models

# In[12]:


models ={
    "Limnear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}


# In[13]:


for name, model in models.items():
    model.fit(x_train, y_train)
    print(f'\n {name} model trained !')


# # Model Evaluation

# In[14]:


results = {}
for name, model in models.items():
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "MSE":mse, "R2 Score": r2} # store result in the dictionary


# In[15]:


results_df = pd.DataFrame(results).T


# In[16]:


results_df


# # Visualizing Results
# * Plot actual vs predicted sales for the best performing model

# In[17]:


best_model_name = results_df['R2 Score'].idxmax()
best_model = models[best_model_name]
y_pred_best = best_model.predict(x_test)


# In[18]:


plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_best, alpha=0.8, color='b')
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)], 'r--', lw=2)
plt.title(f'{best_model_name}- Actual vs Predicted Sales')
plt.xlabel("Actual Sales")
plt.ylabel("Predicated Sales")
plt.grid()
plt.show()


# # Save the Model for Future Predictions

# In[19]:


import joblib


# In[20]:


joblib.dump(best_model,'best_sales_model.pkl')


# In[21]:


print(f"\n{best_model_name} saved as 'best_sales_model.pkl'!")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Serhat Calikoglu
# ### December 12, 2024
# ****

# #### Part 1

# In[181]:

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


























































# In[234]:


s = pd.read_csv("social_media_usage.csv")



# ****
# #### Q2

# In[92]:


def clean_sm(x):
     return np.where(x==1,1,0)
 


# In[94]:


df= pd.DataFrame({ "Column 1": [1,0,3],
                  "Column 2": [1,2,1]})



# In[96]:


clean_sm(df)


# In[98]:


df = clean_sm(df)



# ****

# #### Q3
# 
# Feature Engineering and Exploratory Analysis

# In[ ]:





# In[218]:


ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9, np.nan,s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":clean_sm(s["par"]),
    "married": clean_sm(s["marital"]),
    "female": np.where(s["gender"] == 2,1,0),
    "age": np.where(s["age"] > 98, np.nan, s["age"]),
})




# In[220]:


ss = ss.dropna()



# In[232]:


ss.describe()


# In[169]:


alt.Chart(ss.groupby(["age", "education"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="age",
      y="sm_li",
      color="education:N")


# In[207]:


alt.Chart(ss.groupby(["age", "income"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="age",
      y="sm_li",
      color="income:N")


# In[183]:


sns.scatterplot(ss,
                x="age",
                y="sm_li");


# In[196]:


alt.Chart(ss.groupby(["age", "parent"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="age",
      y="sm_li",
      color="parent:N")


# In[198]:


alt.Chart(ss.groupby(["age", "married"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="age",
      y="sm_li",
      color="married:N")


# In[238]:


alt.Chart(ss.groupby(["age","female"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="age",
      y="sm_li",
      color="female:N")


# In[240]:


alt.Chart(ss.groupby(["income","female"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="income",
      y="sm_li",
      color="female:N")


# In[ ]:


#### Training Model and Evaluation of Performance: Q4, Q5, Q6, Q7, Q8, Q9, Q10


# In[375]:


Y = ss["sm_li"]
X =ss[["income", "education", "parent","married", "female","age"]]


# In[371]:





# In[377]:


X_train, X_test, Y_train, Y_test = train_test_split(X.values,
                                                    Y,
                                                    stratify=Y,      
                                                    test_size=0.2,    
                                                    random_state=986)


# X_train holds 80% of data and has the features used to predict sm_li when training the model. X_test holds 20% of the data with the features is used to test the model on data it has not seen before to evaluate it. Y_ train holds 80% of data and has the target (sm_li) that will be predicted by the features that train the model. Y_test holds 20% of data and has the target and our trained model will predict this target.

# In[379]:


lr = LogisticRegression()


# In[381]:


lr.fit(X_train, Y_train)


# In[383]:


y_pred = lr.predict(X_test)


# In[385]:


pd.DataFrame(confusion_matrix(Y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


# Recall = 40/(40+44) = 0.476
# Precision = 40/(40+23) = 0.635
# F1 = 2 * (0.476 * 0.635) / (0.476 + 0.635) = 0.544

# In[387]:


print(classification_report(Y_test, y_pred))


# In[409]:


newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1,1],
    "female":[1,1],
    "age": [42, 82],
})




# In[389]:





# In[395]:


newdata["prediction_sm_li"] = lr.predict(newdata)


# In[407]:





# In[411]:


# New data for features: age, college, high_income, ideology
person_1 = [8,7,0,1,1,42]
# Predict class, given input features
predicted_class = lr.predict([person_1])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person_1])


print(f"Predicted class: {predicted_class[0]}") # 0=not pro-environment, 1=pro-envronment
print(f"Probability that this uses linked in is: {probs[0][1]}")


# In[415]:


person2 = [8,7,0,1,1,82]
# Predict class, given input features
predicted_class2 = lr.predict([person2])

# Generate probability of positive class (=1)
probs2 = lr.predict_proba([person2])

print(f"Predicted class: {predicted_class2[0]}") # 0=not pro-environment, 1=pro-envronment
print(f"Probability that this uses linked in is: {probs2[0][1]}")



st.markdown("Does this person use Linkedin?")

"Select the options that apply to you."
income = st.selectbox(label="Household Income",
options=("Less than 10,000", 
"$10,000 to $20,000", 
"$20,000 to $30,000", 
"$30,000 to $40,000", 
"$40,000 to $50,000",
"$50,000 to $75,000",
"$75,000 to $100,000",
"$100,000 to $150,000", 
"$150,000+"))

if income == "Less than $10,000":
    income =1
elif income == "$10,000 to $20,000":
    income =2
elif income == "$20,000 to $30,000":
    income =3
elif income == "$30,000 to $40,000":
    income =4
elif income == "$40,000 to $50,000":
    income =5
elif income == "$50,000 to $75,000":
    income =6
elif income == "$75,000 to $100,000":
    income =7
elif income == "$100,000 to $150,000":
    income =8
elif income == "$150,000+":
    income = 9


age = st.slider(label="Enter Your Age", 
        min_value=1,
        max_value=100,
        value=50)

education = st.selectbox(label="Highest Level of Education",
options=("Less than High School",  
"High School Incomplete", 
"High School Graduate", 
"Some college,no degree", 
"Two-year associate degree from college or university", 
"Four-year college or university degree/Bachelor's",
"Some postgraduate or professional schooling, no postgraduate degree",
"Postgraduate or professional degrees, including master's, doctorate, medical or law degree"
))

if education == "Less than High School":
    education = 1
elif education == "High School Incomplete":
    education = 2
elif education == "High School Graduate":
    education = 3
elif education == "Some college,no degree":
    education = 4
elif education == "Two-year associate degree from college or university":
    education = 5
elif education == "Four-year college or university degree/Bachelor's":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree":
    education = 7
elif education == "Postgraduate or professional degrees, including master's, doctorate, medical or law degree":
    education = 8

married = st.selectbox(label="Are you married?",
options=("Yes", 
"No"))
if married == "Yes":
    married = 1
else:
    married = 0

parent = st.selectbox(label="Are you the parent of a child under 18 living in your home?",
options=("Yes", 
"No"))

if parent == "Yes":
    parent = 1
else:
    parent = 0

female = st.selectbox(label="Are you a male or female?",
options=("Female", 
"Male"))

if female == "Yes":
    female = 1
else:
    female = 0






if st.button("Predict"):
    input_data = np.array([[income, education, parent, married, female, age]])
    prob = lr.predict_proba(input_data)
    prediction = "LinkedIn User" if prob[0][1] > 0.5 else "Non-LinkedIn User"
    st.write(f"Prediction: {prediction}")
    st.write(f"Probability: {prob[0][1]:.2f}")
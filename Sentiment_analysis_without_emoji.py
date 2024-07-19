#!/usr/bin/env python
# coding: utf-8

# # Read the Dataset

# In[1]:


import pandas as pd


# In[42]:


data=pd.read_csv('csv_files\\flipkart_dataset.csv')
data


# # Data Preprocessing

# In[3]:


#removing punctuations

import string
def remove_punctuation(text):
    txt_nopunt = "".join([c for c in text if c not in string.punctuation])
    return txt_nopunt
data['comment'] = data['comment'].apply(lambda x: remove_punctuation(x))
print(data['comment'])


# In[4]:


#tokentization
import re
def tokenize(txt):
    tokens=re.split('\W+', txt)
    return tokens
data['comment']= data['comment'].apply(lambda x: tokenize(x.lower()))
print(data)


# In[5]:


# removing stop words

import nltk
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text_tokenized):
    txt_clean=[word for word in text_tokenized if word not in stopwords]
    return txt_clean
data['comment']= data['comment'].apply(lambda x: remove_stopwords(x))

print(data)


# In[6]:


wn=nltk.WordNetLemmatizer()
#lemmatization

def lemmatization(txt):
    lemmed = [wn.lemmatize(word) for word in txt]
    return lemmed

data['comment']=data['comment'].apply(lambda x: lemmatization(x))
print(data)


# In[7]:


#updating the file

# df2=data.rename(columns={0: 'Names', 1: 'Dates', 2: 'Rate', 3: 'Review', 4: 'comment', 5: 'Place'})
# df2.to_csv('csv_files\\preproced_withemoji.csv')


# # Data Labeling

# In[8]:


pip install vaderSentiment


# In[9]:


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_Vader(text):
    over_all_polarity = sid.polarity_scores(text)
    if over_all_polarity['compound'] >= 0.1:
        return "positive"
    else:
        return "negative"
def neg_score(text):
    over_all_polarity = sid.polarity_scores(text)
    return over_all_polarity['neg']
def neu_score(text):
    over_all_polarity = sid.polarity_scores(text)
    return over_all_polarity['neu']
def pos_score(text):
    over_all_polarity = sid.polarity_scores(text)
    return over_all_polarity['pos']
def compound_score(text):
    over_all_polarity = sid.polarity_scores(text)
    return over_all_polarity['compound']

sid = SentimentIntensityAnalyzer()

data_file = pd.read_csv('csv_files/preproced_withemoji.csv')
data_file['pos_score'] = data_file['comment'].apply(lambda x: pos_score(x))
data_file['neg_score'] = data_file['comment'].apply(lambda x: neg_score(x))
data_file['neu_score'] = data_file['comment'].apply(lambda x: neu_score(x))
data_file['compound_score'] = data_file['comment'].apply(lambda x: compound_score(x))
data_file['sentiment_vader'] = data_file['comment'].apply(lambda x: sentiment_Vader(x))

print(data_file)


# In[10]:


# csv_data = data_file.to_csv('csv_files/vadar_withemoji.csv')


# In[11]:


df=data_file


# In[12]:


df


# In[13]:


from sklearn.preprocessing import LabelEncoder

cat_cols=['sentiment_vader']
le=LabelEncoder()
for i in cat_cols:
    df['target']=le.fit_transform(df[i])


# In[14]:


df


# In[15]:


# csv_data = df.to_csv('csv_files\\vadar_2_withoutemoji.csv')


# In[16]:


#df1=pd.read_csv('csv_files\\vadar_2_withemoji.csv')
df1=df


# In[17]:


d=df1['comment']


# # Feature Vectorization

# # Word2Vec 

# In[19]:


from gensim.models import Word2Vec


# In[20]:


import numpy as np
# Train the CBOW model
embedding_dim = 100  # Dimensionality of the word embeddings
window_size = 5  # Context window size
model = Word2Vec(d, window=window_size, sg=0, min_count=1)

# Function to get the feature vector for a review
def get_review_vector(review):
    vectors = [model.wv[word] for word in review if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

# Get feature vectors for all reviews
review_vectors = [get_review_vector(review) for review in d]

# Print the feature vectors
for i, vector in enumerate(review_vectors):
    print(f"Review {i+1} vector: {vector}")


# In[21]:


print(type(review_vectors))
df = pd.DataFrame({"feature_vector": review_vectors})

# Print the DataFrame
print(df)


# # Model
# 
# # AdaBoost
# 

# In[22]:


from sklearn.ensemble import AdaBoostClassifier
# importing train test split funtion
from sklearn.model_selection import train_test_split
#importing metrics module
from sklearn import metrics


# In[23]:


import numpy as np

x=np.vstack(df['feature_vector'].values)
y=df1['target']


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[25]:


#creating adaboost classifier object
ada=AdaBoostClassifier(n_estimators=50,learning_rate=1)
#n_estimators=Number of weak learners to train iteratively
#learning_rate=It contributes to the weights of weak learners.it uses 1 as a default value


# In[26]:


#training the model
model=ada.fit(X_train,y_train)


# In[27]:


#predict the response for test dataset
y_pred=model.predict(X_test)


# In[28]:


from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y_test,y_pred)


# In[29]:


print(accuracy)


# In[31]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate incorrect predictions
incorrect = 1 - accuracy

# Convert accuracy and incorrect values to percentages
accuracy_percentage = accuracy * 100
incorrect_percentage = incorrect * 100

# Create bar graph
labels = ['Correct', 'Incorrect']
values = [accuracy_percentage, incorrect_percentage]
colors = ['blue', 'red']

plt.bar(labels, values, color=colors)

# Add percentage values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Prediction Accuracy')
plt.ylabel('Percentage')
plt.title('Accuracy (without emoji and text + Adaboost)')

# Show the graph
plt.show()


# In[32]:


from sklearn.metrics import precision_score
precision =precision_score(y_test,y_pred)
print('precision:',precision)


# In[33]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(AdaBoost)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # Xgboost

# In[34]:


pip install xgboost


# In[35]:


from xgboost import XGBClassifier
model = XGBClassifier()  
model.fit(X_train , y_train) 


# In[36]:


y_pred = model.predict(X_test)  


# In[37]:


from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y_test,y_pred)
print('Accuracy:',accuracy)


# In[38]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate incorrect predictions
incorrect = 1 - accuracy

# Convert accuracy and incorrect values to percentages
accuracy_percentage = accuracy * 100
incorrect_percentage = incorrect * 100

# Create bar graph
labels = ['Correct', 'Incorrect']
values = [accuracy_percentage, incorrect_percentage]
colors = ['blue', 'red']

plt.bar(labels, values, color=colors)

# Add percentage values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Prediction Accuracy')
plt.ylabel('Percentage')
plt.title('Accuracy (without emoji and text + XGBoost)')

# Show the graph
plt.show()


# In[39]:


from sklearn.metrics import precision_score
precision =precision_score(y_test,y_pred)
print('precision:',precision)


# In[40]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


# Calculate precision score
precision =precision_score(y_test,y_pred)

# Calculate incorrect predictions
incorrect = 1 - precision

# Convert accuracy and incorrect values to percentages
precision_percentage = precision * 100
incorrect_percentage = incorrect * 100

# Create bar graph
labels = ['Correct', 'Incorrect']
values = [precision_percentage, incorrect_percentage]
colors = ['blue', 'red']

plt.bar(labels, values, color=colors)

# Add percentage values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Prediction Accuracy')
plt.ylabel('Percentage')
plt.title('Precision (without emoji and text + XGBoost)')

# Show the graph
plt.show()


# In[45]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(XGBoost)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # FastText

# In[44]:


#https://colab.research.google.com/drive/1xJWdxpT2XQQOZ8sEIQaeDu6eemql5qf4?usp=sharing


# In[ ]:





# # Glove

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


data= pd.read_csv('vadar.csv')

reviews=data['comment']


# In[3]:


vectorizer = CountVectorizer()


# In[4]:


# Fit and transform the reviews into vectors
X = vectorizer.fit_transform(reviews)


# In[5]:


# Convert the vector representation to an array
X = X.toarray()

# Create a DataFrame with a single column for storing the review vectors
df = pd.DataFrame({'ReviewVector': X.tolist()})

# Print the DataFrame
print(df)


# In[6]:


from sklearn.ensemble import AdaBoostClassifier
# importing train test split funtion
from sklearn.model_selection import train_test_split
#importing metrics module
from sklearn import metrics


# In[7]:


import numpy as np
x=np.vstack(df['ReviewVector'].values)

y=data['target']


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[9]:


#creating adaboost classifier object
ada=AdaBoostClassifier(n_estimators=50,learning_rate=1)
#n_estimators=Number of weak learners to train iteratively
#learning_rate=It contributes to the weights of weak learners.it uses 1 as a default value


# In[10]:


#training the model
model=ada.fit(X_train,y_train)


# In[11]:


#predict the response for test dataset
y_pred=model.predict(X_test)


# In[12]:


from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y_test,y_pred)

print(accuracy)


# In[13]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate incorrect predictions
incorrect = 1 - accuracy

# Convert accuracy and incorrect values to percentages
accuracy_percentage = accuracy * 100
incorrect_percentage = incorrect * 100

# Create bar graph
labels = ['Correct', 'Incorrect']
values = [accuracy_percentage, incorrect_percentage]
colors = ['blue', 'red']

plt.bar(labels, values, color=colors)

# Add percentage values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Prediction Accuracy')
plt.ylabel('Percentage')
plt.title('Accuracy (without emoji and text + Adaboost)')

# Show the graph
plt.show()


# In[14]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


# Calculate precision score
precision =precision_score(y_test,y_pred)

# Calculate incorrect predictions
incorrect = 1 - precision

# Convert accuracy and incorrect values to percentages
precision_percentage = precision * 100
incorrect_percentage = incorrect * 100

# Create bar graph
labels = ['Correct', 'Incorrect']
values = [precision_percentage, incorrect_percentage]
colors = ['blue', 'red']

plt.bar(labels, values, color=colors)

# Add percentage values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Prediction Accuracy')
plt.ylabel('Percentage')
plt.title('Precision (without emoji and text + AdaBoost)')

# Show the graph
plt.show()


# In[23]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(AdaBoost)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[16]:


pip install xgboost


# In[17]:


from xgboost import XGBClassifier


# In[18]:


model = XGBClassifier()  
model.fit(X_train , y_train)  
print(model)  


# In[19]:


y_prediction = model.predict(X_test)  


# In[20]:


from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y_test,y_pred)

print(accuracy)


# In[21]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate incorrect predictions
incorrect = 1 - accuracy

# Convert accuracy and incorrect values to percentages
accuracy_percentage = accuracy * 100
incorrect_percentage = incorrect * 100

# Create bar graph
labels = ['Correct', 'Incorrect']
values = [accuracy_percentage, incorrect_percentage]
colors = ['blue', 'red']

plt.bar(labels, values, color=colors)

# Add percentage values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Prediction Accuracy')
plt.ylabel('Percentage')
plt.title('Accuracy (without emoji and text + XGboost)')

# Show the graph
plt.show()


# In[22]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


# Calculate precision score
precision =precision_score(y_test,y_pred)

# Calculate incorrect predictions
incorrect = 1 - precision

# Convert accuracy and incorrect values to percentages
precision_percentage = precision * 100
incorrect_percentage = incorrect * 100

# Create bar graph
labels = ['Correct', 'Incorrect']
values = [precision_percentage, incorrect_percentage]
colors = ['blue', 'red']

plt.bar(labels, values, color=colors)

# Add percentage values on top of bars
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Prediction Accuracy')
plt.ylabel('Percentage')
plt.title('Precision (without emoji and text + XgboostBoost)')

# Show the graph
plt.show()


# In[24]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(XGBoost)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:





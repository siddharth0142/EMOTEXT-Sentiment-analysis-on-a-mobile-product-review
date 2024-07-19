#!/usr/bin/env python
# coding: utf-8

#                                               EMOJI TO TEXT

# # Reading Dataset

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('csv_files\\flipkart_dataset.csv')


# In[3]:


data


# In[4]:


comment=data['comment']


# # Converting Emoji to Text in a review

# In[5]:


pip install emoji


# In[6]:


import emoji


# In[7]:


l=[]
for c in comment:
    k=emoji.demojize(c, delimiters=("", ""))
    l.append(k)
    print(k)
    print(' ')


# In[8]:


l


# In[9]:


data['comment'] = l


# In[10]:


k=[]
for i in l:
    m=i.replace('_', " ")
    k.append(m)


# In[11]:


data['comment'] = k


# In[12]:


data['comment']


# # Data Preprocessing

# In[13]:


#removing punctuations

import string
def remove_punctuation(text):
    txt_nopunt = "".join([c for c in text if c not in string.punctuation])
    return txt_nopunt
data['comment'] = data['comment'].apply(lambda x: remove_punctuation(x))
print(data['comment'])


# In[14]:


#tokentization
import re
def tokenize(txt):
    tokens=re.split('\W+', txt)
    return tokens
data['comment']= data['comment'].apply(lambda x: tokenize(x.lower()))
print(data)


# In[15]:


# removing stop words

import nltk
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text_tokenized):
    txt_clean=[word for word in text_tokenized if word not in stopwords]
    return txt_clean
data['comment']= data['comment'].apply(lambda x: remove_stopwords(x))

print(data)


# In[16]:


wn=nltk.WordNetLemmatizer()
#lemmatization

def lemmatization(txt):
    lemmed = [wn.lemmatize(word) for word in txt]
    return lemmed

data['comment']=data['comment'].apply(lambda x: lemmatization(x))
print(data)


# In[ ]:


#updating the file

# df2=data.rename(columns={0: 'Names', 1: 'Dates', 2: 'Rate', 3: 'Review', 4: 'comment', 5: 'Place'})
# df2.to_csv('csv_files\\preproced_withemoji.csv')


# # Data Labeling

# In[17]:


pip install vaderSentiment


# In[18]:


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


# In[19]:


# csv_data = data_file.to_csv('csv_files/vadar_withemoji.csv')


# In[19]:


df = pd.read_csv('csv_files/vadar_withemoji.csv')


# In[20]:


df


# In[21]:


from sklearn.preprocessing import LabelEncoder

cat_cols=['sentiment_vader']
le=LabelEncoder()
for i in cat_cols:
    df['target']=le.fit_transform(df[i])


# In[22]:


df


# In[24]:


# csv_data = df.to_csv('csv_files\\vadar_2_withemoji.csv')


# In[23]:


import pandas as pd
df1=pd.read_csv('csv_files\\vadar_2_withemoji.csv')
df1


# In[24]:


d=df1['comment']


# # Feature Vectoriztion

# In[ ]:





# # Word2Vec

# In[25]:


from gensim.models import Word2Vec


# In[26]:


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


# In[27]:


x`


# In[28]:


from sklearn.ensemble import AdaBoostClassifier
# importing train test split funtion
from sklearn.model_selection import train_test_split
#importing metrics module
from sklearn import metrics


# In[29]:


import numpy as np

x=np.vstack(df['feature_vector'].values)
y=df1['target']


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# # Modeling

# In[ ]:





# # Adaboost

# In[31]:


#creating adaboost classifier object
ada=AdaBoostClassifier(n_estimators=50,learning_rate=1)
#n_estimators=Number of weak learners to train iteratively
#learning_rate=It contributes to the weights of weak learners.it uses 1 as a default value


# In[32]:


#training the model
model=ada.fit(X_train,y_train)


# In[33]:


#predict the response for test dataset
y_pred=model.predict(X_test)


# In[34]:


from sklearn.metrics import accuracy_score
acc_with_ada_word2vec =accuracy_score(y_test,y_pred)


# In[35]:


print(acc_with_ada_word2vec)


# In[36]:


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
plt.title('Accuracy (with emoji and text + Adaboost)')

# Show the graph
plt.show()


# In[37]:


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
plt.title('Precision (with emoji and text + Adaboost)')

# Show the graph
plt.show()


# In[38]:


from sklearn.metrics import precision_score
precision =precision_score(y_test,y_pred)
print('precision:',precision)


# In[39]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(AdaBoost+word2vec)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # xgboost

# In[40]:


pip install xgboost


# In[41]:


from xgboost import XGBClassifier
model = XGBClassifier()  
model.fit(X_train , y_train) 


# In[43]:


y_pred = model.predict(X_test)  


# In[44]:


from sklearn.metrics import accuracy_score
acc_xgboost_word2vec =accuracy_score(y_test,y_pred)
print('Accuracy:',acc_xgboost_word2vec)


# In[45]:


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
plt.title('Accuracy (with emoji and text + XGBoost)')

# Show the graph
plt.show()


# In[46]:


from sklearn.metrics import precision_score
precision =precision_score(y_test,y_pred)
print('precision:',precision)


# In[47]:


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
plt.title('Precision (with emoji and text + XGBoost)')

# Show the graph
plt.show()


# In[ ]:





# In[48]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(XGBoost+word2vec)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:





# # FastText

# In[ ]:


https://colab.research.google.com/drive/1xJWdxpT2XQQOZ8sEIQaeDu6eemql5qf4?usp=sharing


# In[ ]:





#  

# # Glove

# In[1]:





# In[49]:


data= pd.read_csv('vadar_2_withemoji.csv')


# In[51]:


review_tokens=data['comment']


# In[78]:


import numpy as np
import pandas as pd

# Load pre-trained GloVe word vectors into memory
glove_path = "glove.6B//glove.6B.50d.txt"  # Update with the path to your GloVe vectors file
glove_vectors = {}
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        glove_vectors[word] = vector

# Load your dataset with a text column# Update with the path to your dataset CSV file
text_column = data['comment']

# Vectorize the text column using GloVe vectors
def vectorize_text(text, word_vectors):
    words = text.split()
    vectors = [word_vectors.get(word, np.zeros(100)) for word in words]
    return np.mean(vectors, axis=0)  # Average the vectors to get the text representation

# Create a new column for the text vectors
data["text_vectors"] = text_column.apply(lambda text: vectorize_text(text, glove_vectors))

 # Update with the desired path for the updated dataset


# In[88]:


X=data["text_vectors"]


# In[89]:


# Create a DataFrame with a single column for storing the review vectors
df = pd.DataFrame({'ReviewVector': X.tolist()})

# Print the DataFrame
print(df)


# In[81]:





# In[90]:


from sklearn.ensemble import AdaBoostClassifier
# importing train test split funtion
from sklearn.model_selection import train_test_split
#importing metrics module
from sklearn import metrics


# In[91]:


import numpy as np

x=np.vstack(df['ReviewVector'].values)
y=data['target']


# In[92]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[93]:


#creating adaboost classifier object
ada=AdaBoostClassifier(n_estimators=50,learning_rate=1)
#n_estimators=Number of weak learners to train iteratively
#learning_rate=It contributes to the weights of weak learners.it uses 1 as a default value


# In[94]:


#training the model
model=ada.fit(X_train,y_train)


# In[95]:


#predict the response for test dataset
y_pred=model.predict(X_test)


# In[98]:


from sklearn.metrics import accuracy_score
acc_glove_adaboost =accuracy_score(y_test,y_pred)
print(acc_glove_adaboost)


# In[100]:


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
plt.title('Accuracy (with emoji and text + Adaboost)')

# Show the graph
plt.show()


# In[101]:


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
plt.title('Precision (with emoji and text + AdaBoost)')

# Show the graph
plt.show()


# In[102]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(AdaBoost+glove)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[103]:


pip install xgboost


# In[104]:


from xgboost import XGBClassifier


# In[105]:


model = XGBClassifier()  
model.fit(X_train , y_train)  
print(model)  


# In[106]:


y_prediction = model.predict(X_test)  


# In[108]:


from sklearn.metrics import accuracy_score
acc_glove_xgboost =accuracy_score(y_test,y_pred)

print(acc_glove_xgboost)


# In[109]:


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
plt.title('Accuracy (with emoji and text + XGboost)')

# Show the graph
plt.show()


# In[110]:


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
plt.title('Precision (with emoji and text + XgboostBoost)')

# Show the graph
plt.show()


# In[111]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')



plt.title("Confusion Matrix(XGBoost+glove)")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:





# In[ ]:


acc_with_ada_word2vec=0.8683683683683684

acc_fasttext_adaboost=0.8963963963963963

acc_glove_adaboost=0.7942942942942943


# In[126]:


import matplotlib.pyplot as plt

# Accuracy values for different models or scenarios
scenarios = ['Adaboost+word2vec', 'Adaboost+fasttext', 'Adaboost+GloVe', 'Xgboost+word2vec', 'Xgboost+fasttext', 'Xgboost+GloVe']
accuracies = [0.8683683683683684,0.8963963963963963 ,0.7942942942942943 ,0.9059059059059059,0.8963963963963963 ,0.7942942942942943]  # Replace with your accuracy values
# Create a line graph
plt.figure(figsize=(10, 6))
plt.plot(scenarios, accuracies, marker='o')


plt.ylabel('Accuracy')


# Display gridlines
plt.grid(True)

# Show the plot
plt.show()


# In[ ]:


acc_xgboost_word2vec=0.9059059059059059



acc_fasttext_xgboost=0.8963963963963963



acc_glove_xgboos=0.7942942942942943


# In[114]:


import matplotlib.pyplot as plt

# Accuracy values for different models or scenarios
models = ['Word2Vec', 'Fasttext', 'GloVe']
accuracies = [0.9059059059059059,0.8963963963963963 ,0.7942942942942943]  # Replace with your accuracy values

# Create a line graph
plt.figure(figsize=(8, 6))
plt.plot(models, accuracies, marker='o')

# Set labels and title
plt.xlabel('Feature Vectors')
plt.ylabel('Accuracy')
plt.title('XGboost')

# Display gridlines
plt.grid(True)

# Show the plot
plt.show()


# In[116]:


import matplotlib.pyplot as plt

# Accuracy values for different scenarios
scenarios = ['Adaboost+word2vec', 'Adaboost+fasttext', 'Adaboost+GloVe', 'Xgboost+word2vec', 'Xgboost+fasttext', 'Xgboost+GloVe']
accuracies = [0.8683683683683684,0.8963963963963963 ,0.7942942942942943 ,0.9059059059059059,0.8963963963963963 ,0.7942942942942943]  # Replace with your accuracy values

# Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(scenarios, accuracies, color='blue')

# Set labels and title

plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Scenarios')

# Display values on top of bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


# In[118]:


import matplotlib.pyplot as plt

# Accuracy values for different scenarios
scenarios = ['Adaboost+word2vec', 'Adaboost+fasttext', 'Adaboost+GloVe', 'Xgboost+word2vec', 'Xgboost+fasttext', 'Xgboost+GloVe']
accuracies = [0.8683683683683684,0.8963963963963963 ,0.7942942942942943 ,0.9059059059059059,0.8963963963963963 ,0.7942942942942943]  # Replace with your accuracy values


# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(accuracies, labels=scenarios, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Set title
plt.title('Accuracy Distribution')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





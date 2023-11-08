#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install wordcloud if needed
get_ipython().system('pip install wordcloud')

# Imports 
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


# In[2]:


df = pd.read_csv('D:\stocks_cleaned.csv' ,  names=["date", "tweet"])
df


# In[3]:


# Unify Text from all row
text = df['tweet'].str.cat(sep=' ')
text


# In[124]:


def preprocess_text(text):
    # Download NLTK resources (if not already downloaded)
    nltk.download('punkt')
    nltk.download('stopwords')

    # Load stopwords
    custom_stopwords = set(stopwords.words('english'))

    # Preprocessing steps
    cleaned_tokens = []

    # Data cleaning
    cleaned_text = ' '.join(word for word in text.split() if not word.startswith('#'))
    cleaned_text = cleaned_text.lower()

    # Tokenization and stopword removal
    tokens = word_tokenize(cleaned_text)
    tokens = [word for word in tokens if word.isalpha() and word not in custom_stopwords]

    # Combine all cleaned tokens into a single string
    combined_text = ' '.join(tokens)

    return combined_text


# In[125]:


text = preprocess_text(text)


# In[130]:


# Size of Word Cloud
plt.rcParams["figure.figsize"] = (10,10)

# Make Wordcloud
wordcloud = WordCloud(max_font_size=50, max_words=20, background_color="white"
                      , colormap='flag').generate(str(text))

# Plot Wordcloud
plt.plot()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:





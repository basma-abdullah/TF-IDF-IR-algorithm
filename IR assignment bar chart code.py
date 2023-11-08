#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fetch the "20 Newsgroups" dataset
data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)


# In[19]:


# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data.data)

# Perform information retrieval based on a query
query = "saudi arabia"
query_vector = vectorizer.transform([query])

cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Sort and retrieve relevant documents
results = sorted(zip(cosine_similarities, data.data), key=lambda x: x[0], reverse=True)
top_k = 5  # Retrieve top-K relevant documents
relevant_documents = [doc for _, doc in results[:top_k]]

# Print the relevant documents
print("Relevant Documents:")
for i, doc in enumerate(relevant_documents):
    print(f"Document {i+1}:")
    print(doc)
    print("=" * 50)


# In[21]:


import matplotlib.pyplot as plt

# Retrieve the relevance scores from the sorted results
relevance_scores = [score for score, _ in results[:top_k]]

# Plot the relevance scores
plt.figure(figsize=(8, 5))
plt.bar(range(top_k), relevance_scores)
plt.xticks(range(top_k), ['Document {}'.format(i+1) for i in range(top_k)], rotation=45)
plt.xlabel('Documents')
plt.ylabel('Relevance Score')
plt.title('Relevance Scores - Top {} Documents'.format(top_k))
plt.tight_layout()
plt.show()


# In[ ]:





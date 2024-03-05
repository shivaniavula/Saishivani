#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from CSV file
data = pd.read_csv("BankFAQs.csv")

# Define the vectorizer
vectorizer = CountVectorizer()

# Transform the text data into feature vectors
X = vectorizer.fit_transform(data['Question'])

# Define the labels
y = data['Class']

# Train the SVM model
svm_model = SVC(C=10, kernel='rbf', gamma=0.1, decision_function_shape='ovr')
svm_model.fit(X, y)

# Define the Streamlit app
st.title("Bank FAQ Chatbot")

# Get user input
user_input = st.text_input("Please enter your question:")

# Make predictions
if user_input:
    # Transform the input question into feature vector
    input_vector = vectorizer.transform([user_input])

    # Predict the class of the input question
    predicted_class = svm_model.predict(input_vector)[0]

    # Find the answer of the predicted class that is most similar to the input question
    class_data = data[data['Class'] == predicted_class]
    class_vectors = vectorizer.transform(class_data['Question'])
    similarities = cosine_similarity(input_vector, class_vectors)
    most_similar_index = similarities.argmax()
    predicted_answer = class_data.iloc[most_similar_index]['Answer']

    # Display the predicted class and answer
    st.write("Predicted class:", predicted_class)
    st.write("Predicted answer:", predicted_answer)


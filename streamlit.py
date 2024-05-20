import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
model_path = './final_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Function to predict the label of the input text
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    input_ids = inputs['input_ids']
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

# Streamlit app
st.title("Text Classification with BERT")

# Text input from the user
user_input = st.text_area("Enter text to classify:")

# Button to make the prediction
if st.button("Classify"):
    if user_input:
        prediction = predict(user_input)
        st.write(f"Predicted class: {prediction}")
    else:
        st.write("Please enter some text to classify.")

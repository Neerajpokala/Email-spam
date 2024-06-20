import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the label mapping
label_mapping = {0: "Not Spam", 1: "Spam"}

# Define the inference function
@st.cache_data(persist=True, hash_funcs={RobertaForSequenceClassification: lambda _: None, RobertaTokenizer: lambda _: None})
def inference(_model, _tokenizer, text):
    input_ids = _tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = input_ids.ne(_tokenizer.pad_token_id).float().to(device)
    with torch.no_grad():
        outputs = _model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return prediction

# Streamlit app
st.title("Email Spam Detection App")

# Get user input text
text_input = st.text_area("Enter your text:", height=200)

# Make a prediction when the user clicks the button
if st.button("Predict"):
    if text_input:
        prediction = inference(model, tokenizer, text_input)
        predicted_label = label_mapping[prediction]
        st.success(f"Prediction: {predicted_label}")
    else:
        st.warning("Please enter some text to classify.")
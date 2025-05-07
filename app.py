import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = { 
  1:'Toxic',
  0:'Non Toxic'
}

def main():
    st.title("Fine tuning LLM")
    st.header("Fine tuning LLM with custom dataset")


    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])
    
    st.markdown("<div style='height:300px;'></div>", unsafe_allow_html=True)
    st.markdown(""" \n \n \n \n \n \n \n\n\n\n\n\n
        # Footnote on tech stack
        web framework: https://streamlit.io/ \n
        LLM model: "llama-3-8b-bnb-4bit" \n
        vector store: FAISS (Facebook AI Similarity Search) \n
        Embeddings model: GoogleGenerativeAIEmbeddings(model="models/embedding-001") \n
    """)    

if __name__ == "__main__":
    main()
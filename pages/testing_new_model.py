import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="Test Fine-tuned Model",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("Test Fine-tuned Model")
st.markdown("""
This page allows you to test the fine-tuned model with sample questions.
The model has been trained on psychology Q&A data.
""")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
    return model, tokenizer

try:
    model, tokenizer = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Sample questions
sample_questions = [
    "What is cognitive behavioral therapy?",
    "How does stress affect mental health?",
    "What are the symptoms of anxiety?",
    "How can I improve my mental well-being?",
    "What is the difference between depression and sadness?"
]

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample Questions")
    selected_question = st.selectbox(
        "Choose a sample question or write your own:",
        ["Write your own..."] + sample_questions
    )
    
    if selected_question == "Write your own...":
        user_question = st.text_area("Enter your question:", height=100)
    else:
        user_question = selected_question

with col2:
    st.subheader("Model Response")
    if st.button("Generate Response"):
        if user_question:
            with st.spinner("Generating response..."):
                # Prepare the input
                inputs = tokenizer(user_question, return_tensors="pt", padding=True)
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                
                # Decode and display the response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display the response in a nice format
                st.markdown("### Response:")
                st.write(response)
        else:
            st.warning("Please enter a question first.")

# Add some information about the model
st.markdown("---")
with st.expander("About the Model"):
    st.markdown("""
    ### Model Information
    - Base Model: GPT-2
    - Fine-tuned on: Psychology Q&A dataset
    - Training Method: LoRA (Low-Rank Adaptation)
    
    ### How to Use
    1. Select a sample question or write your own
    2. Click 'Generate Response'
    3. The model will generate a response based on its training
    
    ### Note
    The responses are generated based on the model's training data and may not always be perfect.
    Always consult with professionals for serious psychological concerns.
    """)


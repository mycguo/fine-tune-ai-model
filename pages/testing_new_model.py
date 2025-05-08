import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys

# Disable Streamlit's file watcher
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

# Set page config
st.set_page_config(
    page_title="Compare Models",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("Compare Original vs Fine-tuned Model")
st.markdown("""
This page allows you to compare responses from the original GPT-2 model and our fine-tuned model.
The fine-tuned model has been trained on psychology Q&A data.
""")

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.original_model = None
    st.session_state.original_tokenizer = None
    st.session_state.fine_tuned_model = None
    st.session_state.fine_tuned_tokenizer = None

def load_model_safely(model_name, is_fine_tuned=False):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading {'fine-tuned' if is_fine_tuned else 'original'} model: {str(e)}")
        return None, None

# Load models if not already loaded
if not st.session_state.models_loaded:
    with st.spinner("Loading original model..."):
        original_model, original_tokenizer = load_model_safely("gpt2")
        if original_model is not None:
            st.session_state.original_model = original_model
            st.session_state.original_tokenizer = original_tokenizer
            
            # Try to load fine-tuned model if it exists
            if os.path.exists("fine_tuned_model"):
                with st.spinner("Loading fine-tuned model..."):
                    fine_tuned_model, fine_tuned_tokenizer = load_model_safely("fine_tuned_model", True)
                    if fine_tuned_model is not None:
                        st.session_state.fine_tuned_model = fine_tuned_model
                        st.session_state.fine_tuned_tokenizer = fine_tuned_tokenizer
            
            st.session_state.models_loaded = True
            st.success("Models loaded successfully!")

# Function to generate response
def generate_response(model, tokenizer, question):
    try:
        inputs = tokenizer(question, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=800,
                min_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.92,
                top_k=50,
                do_sample=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.5,
                early_stopping=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=4
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Sample questions
sample_questions = [
    "What is cognitive behavioral therapy?",
    "How does stress affect mental health?",
    "What are the symptoms of anxiety?",
    "How can I improve my mental well-being?",
    "What is the difference between depression and sadness?"
]

# Create input section
st.subheader("Input")
selected_question = st.selectbox(
    "Choose a sample question or write your own:",
    ["Write your own..."] + sample_questions
)

if selected_question == "Write your own...":
    user_question = st.text_area("Enter your question:", height=100)
else:
    user_question = selected_question

# Generate and display responses
if st.button("Generate Responses"):
    if user_question and st.session_state.models_loaded:
        # Create three columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Original GPT-2")
            with st.spinner("Generating original model response..."):
                original_response = generate_response(
                    st.session_state.original_model,
                    st.session_state.original_tokenizer,
                    user_question
                )
                if original_response:
                    st.markdown("### Response:")
                    st.write(original_response)
        
        with col2:
            st.subheader("Fine-tuned Model")
            with st.spinner("Generating fine-tuned model response..."):
                if st.session_state.fine_tuned_model is not None:
                    fine_tuned_response = generate_response(
                        st.session_state.fine_tuned_model,
                        st.session_state.fine_tuned_tokenizer,
                        user_question
                    )
                    if fine_tuned_response:
                        st.markdown("### Response:")
                        st.write(fine_tuned_response)
                else:
                    st.error("Fine-tuned model not available")
        
        with col3:
            st.subheader("Comparison")
            st.markdown("### Key Differences:")
            if st.session_state.fine_tuned_model is not None and original_response and fine_tuned_response:
                # Add some basic comparison metrics
                st.metric("Response Length (Original)", len(original_response.split()))
                st.metric("Response Length (Fine-tuned)", len(fine_tuned_response.split()))
                
                # Add some basic analysis
                st.markdown("#### Analysis:")
                st.write("""
                - The fine-tuned model should provide more focused, psychology-related responses
                - The original model might give more general or off-topic responses
                - Compare the relevance and specificity of the answers
                """)
    else:
        if not st.session_state.models_loaded:
            st.error("Please wait for models to load...")
        else:
            st.warning("Please enter a question first.")

# Add some information about the models
st.markdown("---")
with st.expander("About the Models"):
    st.markdown("""
    ### Model Information
    - Original Model: GPT-2
        - General purpose language model
        - Not specifically trained for psychology
    
    - Fine-tuned Model: GPT-2 + LoRA
        - Base: GPT-2
        - Fine-tuned on: Psychology Q&A dataset
        - Training Method: LoRA (Low-Rank Adaptation)
    
    ### How to Use
    1. Select a sample question or write your own
    2. Click 'Generate Responses'
    3. Compare the responses from both models
    
    ### Note
    The responses are generated based on the models' training data and may not always be perfect.
    Always consult with professionals for serious psychological concerns.
    """)


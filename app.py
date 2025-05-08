import streamlit as st
import openai
import json
import os
from datasets import Dataset
import time

# Set page config
st.set_page_config(
    page_title="Fine-tune GPT-3 for Psychology Q&A",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("Fine-tune GPT-3 for Psychology Q&A")
st.markdown("""
This app allows you to fine-tune a GPT-3 model on psychology Q&A data.
The model will be trained to provide more focused and accurate responses to psychology-related questions.
""")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key
else:
    st.warning("Please enter your OpenAI API key to continue")
    st.stop()

# Function to load and prepare dataset
def load_and_prepare_dataset():
    try:
        with st.spinner("Loading dataset..."):
            # Load dataset from local JSON file
            with open('train.json', 'r') as f:
                data = json.load(f)
            
            # Convert to the chat format expected by GPT-3.5-turbo
            training_data = []
            for item in data:
                training_data.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful psychology assistant."},
                        {"role": "user", "content": item['question']},
                        {"role": "assistant", "content": item['answer']}
                    ]
                })
            
            return training_data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Please make sure train.json exists in the current directory.")
        return None

# Load dataset
dataset = load_and_prepare_dataset()
if dataset is None:
    st.stop()

# Display dataset info
st.subheader("Dataset Information")
st.write(f"Number of examples: {len(dataset)}")
st.write("Example data:")
st.write(dataset[0])

# Model configuration
st.subheader("Model Configuration")
model_name = st.selectbox(
    "Select base model",
    ["gpt-3.5-turbo"],  # Removed GPT-4 as it's slower
    index=0
)

# Training configuration
st.subheader("Training Configuration")
col1, col2 = st.columns(2)

with col1:
    # Reduced epochs to just 1
    n_epochs = 1
    st.write("Epochs: 1 (fixed for fastest training)")
    
    # Increased default batch size
    batch_size = st.slider("Batch size", 4, 32, 16, help="Higher batch size = faster training")

with col2:
    # Reduced subset size range
    subset_size = st.slider("Number of training examples", 5, 50, 20, help="Smaller dataset = faster training")
    st.write("Using a small dataset for quick testing")

# Training section
st.subheader("Start Training")
st.markdown("""
Click the button below to begin the fine-tuning process. 
This is configured for the fastest possible training:
- Using GPT-3.5-turbo
- 1 epoch
- Small dataset
- Large batch size
""")

# Training button
if st.button("Start Training"):
    try:
        with st.spinner("Preparing training data..."):
            # Save training data to a JSONL file
            training_file_path = "training_data.jsonl"
            
            # Use smaller subset
            training_data = dataset[:subset_size]
            
            with open(training_file_path, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            # Upload the training file
            with open(training_file_path, 'rb') as f:
                training_file = openai.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            st.success("Training file uploaded successfully!")
            
            # Start fine-tuning
            with st.spinner("Starting fine-tuning job..."):
                fine_tune_job = openai.fine_tuning.jobs.create(
                    training_file=training_file.id,
                    model=model_name,
                    hyperparameters={
                        "n_epochs": n_epochs,
                        "batch_size": batch_size
                    }
                )
                
                st.info("Fine-tuning job started! This should be much faster now.")
                st.write(f"Job ID: {fine_tune_job.id}")
                
                # Monitor the job status
                while True:
                    job_status = openai.fine_tuning.jobs.retrieve(fine_tune_job.id)
                    st.write(f"Status: {job_status.status}")
                    
                    if job_status.status in ['succeeded', 'failed']:
                        break
                    
                    time.sleep(30)  # Check status every 30 seconds instead of 60
                
                if job_status.status == 'succeeded':
                    st.success("Fine-tuning completed successfully!")
                    st.write(f"Fine-tuned model: {job_status.fine_tuned_model}")
                    
                    # Save the fine-tuned model name to a file
                    with open("fine_tuned_model.txt", "w") as f:
                        f.write(job_status.fine_tuned_model)
                    
                    st.info("You can now use this model in the 'Test Fine-tuned Model' page!")
                else:
                    st.error("Fine-tuning failed!")
                    st.write(f"Error: {job_status.error}")
            
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

# Add some information about the tech stack
st.markdown("---")
with st.expander("About the Tech Stack"):
    st.markdown("""
    ### Technologies Used
    - **Base Model**: GPT-3.5-turbo (OpenAI)
    - **Fine-tuning Method**: OpenAI's Fine-tuning API
    - **Dataset**: Psychology Q&A Dataset
    - **UI**: Streamlit
    
    ### Why GPT-3.5-turbo?
    - Faster training than GPT-4
    - Lower cost
    - Sufficient for most use cases
    
    ### Training Time Optimization
    Current configuration for fastest training:
    - Using GPT-3.5-turbo only
    - Fixed 1 epoch
    - Small dataset (5-50 examples)
    - Large batch size (16-32)
    - 30-second status check intervals
    
    ### Note
    Fine-tuning GPT-3 requires an OpenAI API key and will incur costs based on usage.
    """)
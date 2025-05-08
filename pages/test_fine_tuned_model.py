import streamlit as st
from openai import OpenAI
import json
import os

# Set page config
st.set_page_config(
    page_title="Compare Models",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("Compare Original vs Fine-tuned Model")
st.markdown("""
This page allows you to compare responses from the original GPT-3.5-turbo model and your fine-tuned model.
The fine-tuned model has been trained on psychology Q&A data.
""")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("Please enter your OpenAI API key to continue")
    st.stop()

# Load fine-tuned model name from file if it exists
fine_tuned_model = None
if os.path.exists("fine_tuned_model.txt"):
    with open("fine_tuned_model.txt", "r") as f:
        fine_tuned_model = f.read().strip()
else:
    st.warning("No fine-tuned model found. Please complete the training process first.")
    st.stop()

# Load training data
training_data = []
if os.path.exists("train.json"):
    with open("train.json", "r") as f:
        training_data = json.load(f)

# Load the subset size used during training
subset_size = 20  # Default to 20 if not found
if os.path.exists("training_config.json"):
    with open("training_config.json", "r") as f:
        config = json.load(f)
        subset_size = config.get("subset_size", 20)

# Get only the questions used in training
used_training_data = training_data[:subset_size]

# Display training data
with st.expander("View Training Questions Used", expanded=False):
    st.markdown("### Questions Used in Fine-tuning")
    st.markdown(f"These are the {subset_size} questions that were used to fine-tune the model:")
    
    # Create a search box for filtering questions
    search_query = st.text_input("Search training questions", "")
    
    # Filter questions based on search
    filtered_data = used_training_data
    if search_query:
        filtered_data = [
            item for item in used_training_data 
            if search_query.lower() in item['question'].lower() or 
               search_query.lower() in item['answer'].lower()
        ]
    
    # Display questions in a table format
    if filtered_data:
        for i, item in enumerate(filtered_data, 1):
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Q{i}:** {item['question']}")
                with col2:
                    st.markdown(f"**A{i}:** {item['answer']}")
                st.markdown("---")
        
        st.markdown(f"Showing {len(filtered_data)} of {len(used_training_data)} training examples")
    else:
        st.info("No training data found or no matches for your search.")

# Sample questions from training data
sample_questions = [
    item['question'] for item in used_training_data[:5]
] if used_training_data else [
    "What is cognitive behavioral therapy?",
    "How does stress affect mental health?",
    "What are the symptoms of anxiety?",
    "How can I improve my mental well-being?",
    "What is the difference between depression and sadness?"
]

# Create input section
st.subheader("Ask a Question")
selected_question = st.selectbox(
    "Choose a sample question or write your own:",
    ["Write your own..."] + sample_questions
)

if selected_question == "Write your own...":
    user_question = st.text_area("Enter your question:", height=100)
else:
    user_question = selected_question

# Generate responses
if st.button("Compare Responses"):
    if user_question:
        try:
            # Create two columns for side-by-side comparison
            col1, col2 = st.columns(2)
            
            # Function to generate response
            def generate_response(model_name):
                return client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful psychology assistant."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
            
            # Generate responses in parallel
            with st.spinner("Generating responses..."):
                with col1:
                    st.subheader("Original GPT-3.5-turbo")
                    original_response = generate_response("gpt-3.5-turbo")
                    st.write(original_response.choices[0].message.content)
                    st.markdown("---")
                    st.markdown("### Token Usage:")
                    st.write(f"Prompt tokens: {original_response.usage.prompt_tokens}")
                    st.write(f"Completion tokens: {original_response.usage.completion_tokens}")
                    st.write(f"Total tokens: {original_response.usage.total_tokens}")
                
                with col2:
                    st.subheader("Fine-tuned Model")
                    fine_tuned_response = generate_response(fine_tuned_model)
                    st.write(fine_tuned_response.choices[0].message.content)
                    st.markdown("---")
                    st.markdown("### Token Usage:")
                    st.write(f"Prompt tokens: {fine_tuned_response.usage.prompt_tokens}")
                    st.write(f"Completion tokens: {fine_tuned_response.usage.completion_tokens}")
                    st.write(f"Total tokens: {fine_tuned_response.usage.total_tokens}")
            
            # Add comparison section below
            st.markdown("---")
            st.subheader("Key Differences")
            
            # Calculate response lengths
            original_length = len(original_response.choices[0].message.content.split())
            fine_tuned_length = len(fine_tuned_response.choices[0].message.content.split())
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Response Length", original_length)
            with col2:
                st.metric("Fine-tuned Response Length", fine_tuned_length)
            with col3:
                st.metric("Length Difference", fine_tuned_length - original_length)
            
            # Add some basic analysis
            st.markdown("""
            ### Analysis
            - The fine-tuned model should provide more focused, psychology-related responses
            - The original model might give more general or off-topic responses
            - Compare the relevance and specificity of the answers
            """)
            
        except Exception as e:
            st.error(f"Error generating responses: {str(e)}")
    else:
        st.warning("Please enter a question first.")

# Add some information about the models
st.markdown("---")
with st.expander("About the Models"):
    st.markdown("""
    ### Model Information
    - Original Model: GPT-3.5-turbo
        - General purpose language model
        - Not specifically trained for psychology
    
    - Fine-tuned Model: GPT-3.5-turbo + Custom Training
        - Base: GPT-3.5-turbo
        - Fine-tuned on: Psychology Q&A dataset
        - Optimized for: Psychology-related responses
    
    ### How to Use
    1. Select a sample question or write your own
    2. Click 'Compare Responses'
    3. Compare the responses from both models
    
    ### Note
    The responses are generated based on the models' training data and may not always be perfect.
    Always consult with professionals for serious psychological concerns.
    """) 
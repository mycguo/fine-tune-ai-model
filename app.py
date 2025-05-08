import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import platform
import os
from datasets import Dataset

# Set page config
st.set_page_config(
    page_title="Fine-tune GPT-2 for Psychology Q&A",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("Fine-tune GPT-2 for Psychology Q&A")
st.markdown("""
This app allows you to fine-tune a GPT-2 model on psychology Q&A data using LoRA.
The model will be trained to provide more focused and accurate responses to psychology-related questions.
""")

# Function to load and prepare dataset
def load_and_prepare_dataset():
    try:
        with st.spinner("Loading dataset..."):
            # Load dataset from local JSON file
            import json
            with open('train.json', 'r') as f:
                data = json.load(f)
            
            # Convert to the format expected by the training pipeline
            texts = []
            for item in data:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                texts.append(text)
            
            # Create a simple dataset dictionary
            dataset = {"text": texts}
            
            return dataset
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
st.write(f"Number of examples: {len(dataset['text'])}")
st.write("Example data:")
st.write(dataset['text'][0])

# Model configuration
st.subheader("Model Configuration")
model_name = st.selectbox(
    "Select base model",
    ["gpt2", "gpt2-medium", "gpt2-large"],
    index=0
)

# Training configuration
st.subheader("Training Configuration")
learning_rate = st.slider("Learning rate", 1e-5, 1e-3, 2e-4, format="%.2e")
num_epochs = st.slider("Number of epochs", 1, 5, 1)
batch_size = st.slider("Batch size", 1, 16, 8)
max_length = st.slider("Max sequence length", 128, 512, 256)

# LoRA configuration
st.subheader("LoRA Configuration")
lora_r = st.slider("LoRA rank (r)", 4, 32, 8)
lora_alpha = st.slider("LoRA alpha", 8, 64, 16)
lora_dropout = st.slider("LoRA dropout", 0.0, 0.5, 0.1)

# Training arguments
training_args = TrainingArguments(
    output_dir="fine_tuned_model",
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    max_steps=100,
    logging_steps=5,
    save_steps=50,
    save_total_limit=1,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    gradient_accumulation_steps=2,
    warmup_steps=10,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    label_names=["labels"]  # Explicitly set label names
)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        with st.spinner("Loading model and tokenizer..."):
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            # Set the model's maximum sequence length
            model.config.max_position_embeddings = max_length
            return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, tokenizer = load_model_and_tokenizer()
if model is None or tokenizer is None:
    st.stop()

# Configure LoRA
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Convert our data to a HuggingFace Dataset
train_dataset = Dataset.from_dict({"text": dataset["text"]})

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # For causal language modeling, the labels are the same as the input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Apply tokenization to the dataset
tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args
)

# Training section
st.subheader("Start Training")
st.markdown("Click the button below to begin the fine-tuning process. This will train the model on the psychology Q&A dataset using the configured parameters.")

# Training button
if st.button("Start Training"):
    try:
        with st.spinner("Training in progress..."):
            trainer_stats = trainer.train()
            
            # Display training metrics
            st.subheader("Training Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Loss", f"{trainer_stats.training_loss:.4f}")
            with col2:
                st.metric("Steps", f"{trainer_stats.global_step}")
            
            # Save the model
            trainer.save_model()
            tokenizer.save_pretrained("fine_tuned_model")
            
            # Create a zip file of the saved model
            import shutil
            shutil.make_archive("fine_tuned_model", 'zip', "fine_tuned_model")
            
            # Display download button
            with open("fine_tuned_model.zip", "rb") as file:
                st.download_button(
                    label="Download Fine-tuned Model",
                    data=file,
                    file_name="fine_tuned_model.zip",
                    mime="application/zip"
                )
            
            st.success("Training completed successfully! Model has been saved.")
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

# Add some information about the tech stack
st.markdown("---")
with st.expander("About the Tech Stack"):
    st.markdown("""
    ### Technologies Used
    - **Base Model**: GPT-2 (Hugging Face Transformers)
    - **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
    - **Dataset**: Psychology Q&A Dataset
    - **Framework**: PyTorch
    - **UI**: Streamlit
    
    ### Why LoRA?
    LoRA is an efficient fine-tuning method that:
    - Reduces memory usage
    - Speeds up training
    - Maintains model quality
    - Allows for easy model switching
    """)
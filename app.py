import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import pprint
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
import platform

max_seq_length = 2048

# Check if we're on macOS
is_mac = platform.system() == "Darwin"

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float32,  # Always use float32
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "c_attn",  # GPT-2's attention module
        "c_proj",  # GPT-2's output projection
        "c_fc",    # GPT-2's feed-forward up projection
        "c_proj"   # GPT-2's feed-forward down projection
    ],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

chat_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instruction = ""
    inputs = examples["question"]
    outputs = examples["answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = chat_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("BoltMonkey/psychology-question-answer", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)



trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=False,  # Disable fp16
        bf16=False,  # Disable bf16
        logging_steps=1,
        optim="adamw_torch",  # Use standard AdamW
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
    formatting_func=lambda x: x["text"]
)

def main():
    st.title("Fine tuning LLM")
    st.header("Fine tuning LLM with custom dataset")

    button1 = st.button("Click the button to start training")
    if button1:
        with st.spinner('Training in progress...'):
            trainer_stats = trainer.train()
            
            # Display training metrics
            st.subheader("Training Statistics")
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Loss", f"{trainer_stats.training_loss:.4f}")
                
            with col2:
                st.metric("Steps", f"{trainer_stats.global_step}")
            
            # Display detailed metrics in an expander
            with st.expander("Detailed Training Metrics"):
                metrics_dict = {
                    "training_loss": trainer_stats.training_loss,
                    "global_step": trainer_stats.global_step
                }
                st.json(metrics_dict)
            
            # Save the model
            save_path = "fine_tuned_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Create a zip file of the saved model
            import shutil
            shutil.make_archive("fine_tuned_model", 'zip', save_path)
            
            # Display download button
            with open("fine_tuned_model.zip", "rb") as file:
                st.download_button(
                    label="Download Fine-tuned Model",
                    data=file,
                    file_name="fine_tuned_model.zip",
                    mime="application/zip"
                )
            
            # Display a success message
            st.success("Training completed successfully! Model has been saved.")
    
    st.markdown("<div style='height:300px;'></div>", unsafe_allow_html=True)
    st.markdown(""" \n \n \n \n \n \n \n\n\n\n\n\n
        # Footnote on tech stack
        web framework: https://streamlit.io/ \n
        LLM model: "gpt2" \n
    """)    

    # Display example data in Streamlit
    st.subheader("Example Data")
    st.write("Sample 250:")
    st.json(dataset[250])
    st.write("Sample 260:")
    st.json(dataset[260])

if __name__ == "__main__":
    main()
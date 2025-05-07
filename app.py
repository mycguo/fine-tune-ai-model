import streamlit as st
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pprint
from trl import SFTTrainer
from transformers import TrainingArguments



max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

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
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = chat_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = load_dataset("BoltMonkey/psychology-question-answer", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

#Here are a few examples of what the data looks like
pprint.pprint(dataset[250])
pprint.pprint(dataset[260])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

def main():
    st.title("Fine tuning LLM")
    st.header("Fine tuning LLM with custom dataset")


    button1 = st.button("Click the button to start training")
    if button1:
        trainer_stats = trainer.train()
        st.write(trainer_stats)
    
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
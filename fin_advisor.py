import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
import time

# Load dataset
df = pd.read_csv("E:/Datasets/Finance_data.csv")  # Adjust path as needed

def create_input(row):
    profile = (
        f"User Profile:\n"
        f"- Gender: {row['gender']}\n"
        f"- Age: {row['age']}\n"
        f"- Investment Avenues Interested: {row['Investment_Avenues']}\n"
        f"- Preferred Investments:\n"
    )
    investments = []
    investment_columns = ['Mutual_Funds', 'Equity_Market', 'Debentures', 'Government_Bonds',
                          'Fixed_Deposits', 'PPF', 'Gold']
    for col in investment_columns:
        preference = row[col]
        investments.append(f"  - {col.replace('_', ' ')} (Preference: {preference})")
    profile += '\n'.join(investments) + '\n'
    profile += (
        f"- Investment Objectives: {row['Objective']}\n"
        f"- Investment Purpose: {row['Purpose']}\n"
        f"- Investment Duration: {row['Duration']}\n"
        f"- Expected Returns: {row['Expect']}\n"
        f"- Savings Objective: {row['What are your savings objectives?']}\n"
        f"- Source of Information: {row['Source']}\n\n"
        f"Question:\n"
        f"What investment strategies should I consider?"
    )
    return profile

def create_output(row):
    output = (
        f"Considering your objectives of {row['Objective']} and {row['Purpose']} over {row['Duration']}, "
        f"you might explore investment avenues like {row['Avenue']}. "
        f"Given your expected returns of {row['Expect']}, these options align with your goals. "
        f"Remember to diversify your portfolio and assess the risks involved. "
        f"Consulting a financial advisor can provide personalized guidance."
    )
    return output

# Create input and output columns
df['input'] = df.apply(create_input, axis=1)
df['output'] = df.apply(create_output, axis=1)

# Combine input and output into text column
df['text'] = df.apply(lambda row: f"input: {row['input']}\noutput: {row['output']}", axis=1)

# Remove rows with missing values
df.dropna(subset=['input', 'output', 'text'], inplace=True)

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Display an example
example_row = df.iloc[0]
print("Input:")
print(example_row['input'])
print("\nOutput:")
print(example_row['output'])

# Create Dataset
train_data = Dataset.from_pandas(df[['text']])

# Define a formatting function for SFTTrainer
def formatting_prompts_func(example):
    return f"input: {example['text'].split('input: ')[1].split('output: ')[0]}output: {example['text'].split('output: ')[1]}"

# Initialize Model
model_name = "distilgpt2"
compute_dtype = torch.float32  # Use float32 for CPU stability

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 256  # Further reduced for CPU
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)

# Set padding token for distilgpt2
tokenizer.pad_token = tokenizer.eos_token

# Fine-Tune the Model
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"],  # Correct for distilgpt2
)

training_arguments = TrainingArguments(
    output_dir="financial_advisor_model",
    num_train_epochs=1,  # Single epoch for testing
    per_device_train_batch_size=1,  # Reduced to minimize memory usage
    gradient_accumulation_steps=4,  # Reduced for faster steps
    optim="adamw_torch",
    save_steps=0,
    logging_steps=1,  # Frequent logging to monitor progress
    learning_rate=5e-4,
    weight_decay=0.001,
    fp16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=peft_config,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
)

# Start training with timeout check
print("Starting training...")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Save the fine-tuned model and tokenizer
trainer.save_model()
tokenizer.save_pretrained("./financial_advisor_model")

# Load and merge the fine-tuned model
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./financial_advisor_model",
    torch_dtype=compute_dtype,
    device_map=None,  # CPU
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("./financial_advisor_pretrained", safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained("./financial_advisor_pretrained")

# Load the pretrained model for inference
model = AutoModelForCausalLM.from_pretrained(
    "financial_advisor_pretrained",
    torch_dtype=compute_dtype,
)

device = torch.device('cpu')
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("financial_advisor_pretrained")

def get_financial_advice(user_profile, model=model, tokenizer=tokenizer):
    prompt = f"input: {user_profile}\noutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split('output:')[-1].strip()

# Example usage
example_input = df.iloc[0]['input']
print("User Profile and Question:")
print(example_input)
print("\nGenerated Advice:")
print(get_financial_advice(example_input))
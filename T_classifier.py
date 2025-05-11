# t_classifier_finetune.py

import os
import torch
from datetime import datetime
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

# ─── 0) Environment & caching ────────────────────────────────────────────────
os.environ["HF_HOME"] = "/lustre/isaac24/proj/UTK0285/jshao2/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_TOKEN"] = "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"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ─── 1) Configurable variables ───────────────────────────────────────────────
PROJECT_NAME           = "tnmLlama3_T14"
DATASET_FILE           = "T_classifier.csv"
BASE_MODEL             = "meta-llama/Meta-Llama-3-8B-Instruct"
TRAIN_TEST_SPLIT_RATIO = 0.1
PREDICTOR_COL          = "text"
RESPONSE_COL           = "t"

# ─── 2) Load & quantize the base Llama model ─────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# ─── 3) Load & preprocess your CSV into an HF Dataset ─────────────────────────
df = pd.read_csv(DATASET_FILE).dropna(subset=[PREDICTOR_COL])
hf_dataset = Dataset.from_pandas(df)

# ─── 4) Dump 5 raw examples ───────────────────────────────────────────────────
os.makedirs(f"{PROJECT_NAME}_results", exist_ok=True)
with open(f"{PROJECT_NAME}_results/dataset_samples.txt", "w") as f:
    for i, ex in enumerate(hf_dataset.select(range(5))):
        prompt = ex[PREDICTOR_COL]
        response = str(ex[RESPONSE_COL])
        f.write(f"=== Example {i+1} ===\nPrompt:\n{prompt}\nResponse:\n{response}\n\n")

# ─── 5) Pre-fine-tuning generations ───────────────────────────────────────────
pipe_pre = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
with open(f"{PROJECT_NAME}_results/model_usage.txt", "w") as f:
    f.write("=== MODEL OUTPUT BEFORE FINE-TUNING ===\n\n")
    for i, ex in enumerate(hf_dataset.select(range(5))):
        prompt = ex[PREDICTOR_COL]
        chat = tokenizer.apply_chat_template([
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": prompt},
        ], tokenize=False)["text"]
        gen = pipe_pre(chat, max_new_tokens=64)[0]["generated_text"]
        f.write(f"--- Prompt {i+1} ---\n{prompt}\n--- Generated ---\n{gen}\n\n")

# ─── 6) Chat-format & split ──────────────────────────────────────────────────
def format_chat(ex):
    convo = [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": ex[PREDICTOR_COL]},
        {"role": "assistant", "content": str(ex[RESPONSE_COL])},
    ]
    return {"text": tokenizer.apply_chat_template(convo, tokenize=False)["text"]}

hf_dataset = hf_dataset.map(format_chat, batched=False)
split_ds = hf_dataset.train_test_split(test_size=TRAIN_TEST_SPLIT_RATIO)
train_ds = split_ds["train"]
val_ds   = split_ds["test"]

# ─── 7) Tokenization for SFT ─────────────────────────────────────────────────
def tokenize_fn(ex):
    enc = tokenizer(
        ex["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_train = train_ds.map(tokenize_fn, batched=False)
tokenized_val   = val_ds.map(tokenize_fn, batched=False)

# ─── 8) Prepare for k-bit LoRA training ───────────────────────────────────────
model.train()
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ─── 9) SFT TrainingArguments & Trainer ──────────────────────────────────────
sft_config = SFTConfig(
    output_dir=f"{PROJECT_NAME}_results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=500,
    logging_steps=50,
    learning_rate=4e-5,
    bf16=True,
    fp16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=sft_config,
)

# ─── 10) Train & save ─────────────────────────────────────────────────────────
trainer.train()
model.save_pretrained(f"{PROJECT_NAME}_lora")

# ─── 11) Post-fine-tuning generations ────────────────────────────────────────
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
base.config.use_cache = False
finetuned = PeftModel.from_pretrained(base, f"{PROJECT_NAME}_lora", device_map="auto", torch_dtype=torch.bfloat16)

pipe_post = pipeline("text-generation", model=finetuned, tokenizer=tokenizer, device_map="auto")
with open(f"{PROJECT_NAME}_results/model_usage.txt", "a") as f:
    f.write("\n=== MODEL OUTPUT AFTER FINE-TUNING ===\n\n")
    for i, ex in enumerate(hf_dataset.select(range(5))):
        prompt = ex[PREDICTOR_COL]
        chat = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ], tokenize=False)["text"]
        gen = pipe_post(chat, max_new_tokens=64)[0]["generated_text"]
        f.write(f"--- Prompt {i+1} ---\n{prompt}\n--- Generated ---\n{gen}\n\n")

print("✅ Fine-tuning pipeline complete!")

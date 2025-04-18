import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from accelerate import Accelerator
from PIL import Image
import wandb
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from transformers.trainer_utils import EvalPrediction
import re

os.environ["WANDB_DISABLED"] = "true"


best_adapter_checkpoint = './llama_ft' 
TEMPERATURE = 0.7 
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_SAMPLE_SIZE = 0.25  
LLAMA_MODEL_HF_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct" 


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16 
)

peft_config = LoraConfig(
        lora_alpha=16, 
        lora_dropout=0.05, 
        r=8, 
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="llama_ft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    seed=42,
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="text",  
    dataset_kwargs={"skip_prepare_dataset": True},
    max_seq_length=1024,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  
    logging_dir="llama3.2/logs",
    report_to="none"  
)
args.remove_unused_columns=False


def format_data(sample):
    print("found the image path ", sample["image"])
    image_path = "Datasets/" + sample["image"]

    if image_path is None:
        raise FileNotFoundError(f"No image found for ID {sample['unique_id']} with any of the expected extensions.")

    image = Image.open(image_path).convert("RGB")
    max_size = (224, 224)  
    image.thumbnail(max_size, Image.Resampling.LANCZOS)  
    return {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{sample['content']}\n\n"
                    ),
                },
                {
                    "type": "image",
                    "image": image,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"This image and text pair should be classified as: {'Likely (1)' if sample['multimodal_label'] == 1 else 'Unlikely (0)'}"}],
        }
    ]
}
full_df = pd.read_csv('Benchmarking/sample_train.csv')

test_df = pd.read_csv('Benchmarking/sample_test.csv')

train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

for df in [train_df, val_df, test_df]:
    df['multimodal_label'] = df['multimodal_label'].map({'Likely': 1, 'Unlikely': 0})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

def process_dataset(dataset, desc):
    for sample in tqdm(dataset, desc=desc):
        try:
            yield format_data(sample)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

train_dataset_processed = list(process_dataset(train_dataset, "Processing Train Dataset"))
val_dataset_processed = list(process_dataset(val_dataset, "Processing Val Dataset"))
test_dataset_processed = list(process_dataset(test_dataset, "Processing Test Dataset"))

if train_dataset_processed:
    print("\n\nTraining Data Example:")
    print(train_dataset_processed[0])
else:
    print("No processed training samples available.")
    

model = AutoModelForVision2Seq.from_pretrained(
    LLAMA_MODEL_HF_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(LLAMA_MODEL_HF_ID)

def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [example["messages"][0]['content'][1]['image'] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens: labels[labels == image_token_id] = -100

    batch["labels"] = labels

    return batch

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset_processed,
    eval_dataset=val_dataset_processed,  
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)
torch.cuda.empty_cache()

trainer.train()

merged_model = trainer.model.merge_and_unload()

output_dir = args.output_dir  
merged_model.save_pretrained(output_dir, safe_serialization=False)

print(f"Merged model saved at {output_dir}")
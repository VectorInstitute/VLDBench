import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from accelerate import Accelerator
from PIL import Image
import wandb
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from transformers.trainer_utils import EvalPrediction
import re

os.environ["WANDB_DISABLED"] = "true"


best_adapter_checkpoint = './pixtral_ift' 
TEMPERATURE = 0.7 
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_SAMPLE_SIZE = 0.25  
PIXTRAL_MODEL_HF_ID = "/model-weights/Pixtral-12B-2409/" 


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
    output_dir="pixtral_ift",
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
    logging_dir="pixtral/logs",
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
                        "Assess the text and image below for potential disinformation (try finding deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.\n"
                        "If you find any of the listed rhetorical techniques below, then the article is likely disinformation; if not, it is likely not disinformation.\n\n"
                        "Rhetorical Techniques Checklist:\n"
                        "- Emotional Appeal: Uses language or imagery that intentionally invokes extreme emotions like fear or anger, aiming to distract from lack of factual backing.\n"
                        "- Exaggeration and Hyperbole: Makes claims that are unsupported by evidence, or presents normal situations as extraordinary to manipulate perceptions.\n"
                        "- Bias and Subjectivity: Presents information in a way that unreasonably favors one perspective, omitting key facts that might provide balance.\n"
                        "- Repetition: Uses repeated messaging of specific points or misleading statements to embed a biased viewpoint in the reader's mind.\n"
                        "- Specific Word Choices: Employs emotionally charged or misleading terms to sway opinions subtly, often in a manipulative manner.\n"
                        "- Appeals to Authority: References authorities who lack relevant expertise or cites sources that do not have the credentials to be considered authoritative in the context.\n"
                        "- Lack of Verifiable Sources: Relies on sources that either cannot be verified or do not exist, suggesting a fabrication of information.\n"
                        "- Logical Fallacies: Engages in flawed reasoning such as circular reasoning, strawman arguments, or ad hominem attacks that undermine logical debate.\n"
                        "- Conspiracy Theories: Propagates theories that lack proof and often contain elements of paranoia or implausible scenarios as facts.\n"
                        "- Inconsistencies and Factual Errors: Contains multiple contradictions or factual inaccuracies that are easily disprovable, indicating a lack of concern for truth.\n"
                        "- Selective Omission: Deliberately leaves out crucial information that is essential for a fair understanding of the topic, skewing perception.\n"
                        "- Manipulative Framing: Frames issues in a way that leaves out alternative perspectives or possible explanations, focusing only on aspects that support a biased narrative.\n"
                        "- Visual Manipulation: Use of edited or doctored images that distort reality or present misleading visual information.\n"
                        "- Misleading Captions: Pairing images with captions that misrepresent or skew the context of what is shown.\n"
                        "- Emotional Imagery: Using visually striking or emotionally charged images to evoke strong reactions, potentially distracting from factual content.\n"
                        "- Selective Framing: Cropping or framing images in ways that exclude important context or create a misleading impression.\n"
                        "- Juxtaposition: Placing unrelated images side-by-side to imply a false connection or narrative.\n\n"
                        "When examining the image, consider:\n"
                        "- Does it appear to be manipulated or doctored?\n"
                        "- Is the image presented in a context that matches its actual content?\n"
                        "- Are there any visual elements designed to provoke strong emotional responses?\n"
                        "- Is the framing or composition of the image potentially misleading?\n"
                        "- Does the image reinforce or contradict the claims made in the text?\n\n"
                        "Evaluate how the text and image work together. Are they consistent, or does one contradict or misrepresent the other? Does the combination of text and image create a misleading impression?\n\n"
                        "Remember that bias can be subtle. Look for nuanced ways the image might reinforce stereotypes, present a one-sided view, or appeal to emotions rather than facts.\n\n"
                        f"{sample['content']}\n\n"
                        "Respond ONLY with the classification 'Likely (1)' or 'Unlikely (0)' without any additional explanation."
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
full_df = pd.read_csv('Benchmarking/2sample_train.csv')

test_df = pd.read_csv('Benchmarking/2sample_test.csv')

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
    

model = LlavaNextForConditionalGeneration.from_pretrained(
    PIXTRAL_MODEL_HF_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(PIXTRAL_MODEL_HF_ID)

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
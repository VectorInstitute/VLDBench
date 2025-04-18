import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import pandas as pd
import os
from PIL import Image
import re, base64


# --------------------
# Configuration
# --------------------
# Path to the merged multimodal model
merged_model_save_path = 'meta-llama/Llama-3.2-11B-Vision-Instruct'

# Path to your dataset CSV and image folder
dataset_csv_path = 'llama-vision-ift-inference_results.csv'
image_folder_path = 'dataset-phi/sampled_images'

# Output file
output_csv_path = 'judge-llama-ift-vision-results.csv'

# --------------------
# Model and Processor Setup
# --------------------
# Load quantization config if used during saving
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the merged model
model = AutoModelForVision2Seq.from_pretrained(
    merged_model_save_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    quantization_config=bnb_config
)
model.eval()

# Load the processor
processor = AutoProcessor.from_pretrained(merged_model_save_path)

# --------------------
# Data Loading
# --------------------
# Load the main DataFrame
df = pd.read_csv(dataset_csv_path)

# --------------------
# Utility Functions
# --------------------

def prepare_samples_from_dataframe(df, image_folder_path):
#     """ Prepares samples by combining text, image, and predicted labels. """
    samples = []
    
    for index, row in df.iterrows():
        unique_id = row['unique_id']
        text_content = row['first_paragraph']
        
        # Find the image file
        possible_extensions = ['jpg', 'jpeg', 'png']
        image_path = None
        
        for ext in possible_extensions:
            img_path = os.path.join(image_folder_path, f"{unique_id}.{ext}")
            if os.path.exists(img_path):
                image_path = img_path
                break
        
        if image_path is None:
            print(f"No image found for ID {unique_id}")
            continue
        
        # Open and resize the image
        image = Image.open(image_path).convert("RGB")
        max_size = (224, 224)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Define the disinformation checklist
        disinformation_definition = (
            "Disinformation can be identified by the use of certain rhetorical and visual techniques that aim to mislead or manipulate. "
            "Below is a checklist of common techniques used in disinformation:\n\n"
            "- Emotional Appeal: Uses language or imagery to provoke strong emotions (e.g., fear, anger) to distract from factual accuracy.\n"
            "- Exaggeration and Hyperbole: Makes unsupported claims or presents ordinary situations as extraordinary to manipulate perceptions.\n"
            "- Bias and Subjectivity: Unfairly favors one perspective while omitting key facts that would provide balance.\n"
            "- Repetition: Repeats misleading statements or biased points to embed them in the audience's mind.\n"
            "- Manipulative Word Choices: Uses emotionally charged or misleading terms to subtly influence opinions.\n"
            "- Appeals to Unqualified Authority: Cites sources or authorities lacking relevant expertise in the context.\n"
            "- Lack of Verifiable Sources: Relies on unverifiable or fabricated sources to support claims.\n"
            "- Logical Fallacies: Engages in flawed reasoning (e.g., circular logic, personal attacks) that undermines logical debate.\n"
            "- Conspiracy Theories: Promotes unproven, implausible theories often rooted in paranoia.\n"
            "- Inconsistencies and Factual Errors: Contains contradictions or easily disprovable inaccuracies, showing disregard for truth.\n"
            "- Selective Omission: Deliberately leaves out crucial information that would provide a more balanced understanding.\n"
            "- Manipulative Framing: Presents issues in a way that excludes alternative perspectives, focusing only on aspects that support a biased narrative.\n"
            "- Visual Manipulation: Uses edited or misleading images, staged scenarios, or out-of-context visuals to create false narratives.\n"
            "- Misleading Captions: Provides captions that misrepresent the content of an image, leading to false interpretations.\n"
            "- Contextual Mismatch: Pairs images with unrelated text to suggest false connections between unrelated events."
        )

        predicted_label = 1
        sample_text = (
            f"Evaluate if the following text and image pair is likely disinformation based on the predicted label.\n\n"
            f"Definition of Disinformation:\n{disinformation_definition}\n\n"
            f"Input Text: {text_content}\n\n"
            f"Predicted Label: {'Disinformation' if predicted_label == 1 else 'Not Disinformation'}\n\n"
            f"Unless there are strong reasons not to agree with this prediction, please confirm agreement. "
            f"Do you agree with this prediction? Please respond with '[Agree]' or '[Not Agree]' only."
        )

        sample = {
            "unique_id": unique_id,
            "predicted_label": 'Disinformation' if predicted_label == 1 else 'Not Disinformation',
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample_text},
                        {"type": "image", "image": image}
                    ]
                }
            ]
        }
        
        samples.append(sample)
    
    return samples

def extract_assistant_reply(generated_text):
    """ Extracts and cleans up the assistant's reply from the generated text. """
    lines = generated_text.strip().split('\n')
    for idx, line in enumerate(lines):
        if line.strip().lower() == 'assistant':
            assistant_reply = '\n'.join(lines[idx+1:]).strip()
            
            # Clean up any verbose content after first occurrence of 'Agree' or 'Not Agree'
            match = re.search(r'\b(Agree|Not Agree|Yes|No|Correct|Incorrect|I agree|I do not agree)\b', assistant_reply, re.IGNORECASE)
            if match:
                return match.group(0).capitalize()
            print(f"Assistant's Full Reply: {assistant_reply}")
            return assistant_reply
    
    return lines[-1].strip()


def extract_classification(assistant_reply):
    """ Extracts the classification ('Agree' or 'Not Agree') from the assistant's reply. """
    match = re.search(r'\b(Agree|Not Agree|Yes|No|Correct|Incorrect)\b', assistant_reply, re.IGNORECASE)
    if match:
        return 'Agree' if match.group(1).lower() in ['agree', 'yes', 'correct'] else 'Not Agree'
    else:
        return 'Unknown'


def generate_prediction(sample):
    """ Generates a prediction using the model and processor. """
    texts = processor.apply_chat_template(sample["messages"], tokenize=False)
    image_input = sample["messages"][0]['content'][1]['image']
    
    inputs = processor(text=texts, images=image_input, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Increase max_new_tokens to allow longer responses
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Increased from 50 to 100
            do_sample=False,
            num_beams=1,
        )
    
    generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract and classify response
    assistant_reply = extract_assistant_reply(generated_texts[0])
    classification = extract_classification(assistant_reply)
    
    return assistant_reply, classification

# --------------------
# Sample Preparation and Inference
# --------------------
samples = prepare_samples_from_dataframe(df, image_folder_path)

# Optionally limit to a subset for testing purposes (e.g., first 900 samples)
samples_to_infer = samples[:900]

results = []
for sample in samples_to_infer:
    assistant_reply, classification = generate_prediction(sample)
    
    result = {
        'unique_id': sample['unique_id'],
        'predicted_label': sample['predicted_label'],
        'assistant_reply': assistant_reply,
        'classification': classification
    }
    
    results.append(result)

    # Optional: Print progress for each sample processed.
    print(f"Sample ID: {sample['unique_id']}")
    print("Assistant's Reply:")
    print(assistant_reply)
    print(f"Classification: {classification}")
    print("-" * 50)

# --------------------
# Saving Results
# --------------------
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
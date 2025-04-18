import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Features, Value, Image as HFImage
import json
import time
from argparse import ArgumentParser
from tqdm import tqdm
import os
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# Define the features of the dataset
features = Features({
    "unique_id": Value("string"),
    "outlet": Value("string"),
    "headline": Value("string"),
    "image_description": Value("string"),
    "article_text": Value("string"),
    "image": HFImage(),
    "date_published": Value("string"),
    "source_url": Value("string"),
    "canonical_link": Value("string"),
    "new_categories": Value("string"),
    "news_categories_confidence_scores": Value("string"),
    "text_label": Value("string"),
    "multimodal_label": Value("string"),
})

# Load the Hugging Face dataset
def load_hf_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Dataset size: {len(dataset)} entries")
    return dataset

# Function to truncate content to 2000 words
def truncate_to_2000_words(text):
    cleaned_text = ' '.join(text.split())
    words = cleaned_text.split()
    return ' '.join(words[:2000]) if len(words) > 2000 else cleaned_text

# Collate function
def collate_fn(batch):
    images = [item['image'] for item in batch]
    article_texts = [item['article_text'] for item in batch]
    labels = [item['multimodal_label'] for item in batch]
    ids = [item['unique_id'] for item in batch]

    return {
        'images': images,
        'article_texts': article_texts,
        'answers': labels,
        'ids': ids
    }

# Evaluate function
def evaluate(model, dataloader, processor, device):
    model.eval()
    results = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="batch") as pbar:
            for batch in dataloader:
                images = batch['images']
                article_texts = batch['article_texts']
                labels = batch['answers']
                ids = batch['ids']

                for img, text, label, id in zip(images, article_texts, labels, ids):
                    try:
                        text = truncate_to_2000_words(text)

                        prompt = (
                            "Assess the text and image below for potential disinformation.\n"
                            "Return only 'Classification: Likely' or 'Classification: Unlikely'.\n\n"
                            f"Article: {text}\n\n"
                        )
                    
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": str(prompt)},
                                    {"type": "image"},
                                ],
                            },
                        ]

                        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

                        # Generate output
                        output = model.generate(**inputs, max_new_tokens=50)
                        predicted_answer = processor.decode(output[0], skip_special_tokens=True)

                        # Post-process output
                        if "Classification: Likely" in predicted_answer:
                            predicted_answer = "Classification: Likely"
                        elif "Classification: Unlikely" in predicted_answer:
                            predicted_answer = "Classification: Unlikely"
                        else:
                            predicted_answer = "Classification: Unknown"

                        results.append({
                            "id": id,
                            "predicted_answer": predicted_answer,
                            "ground_truth": label,
                            "prompt": prompt
                        })
                    
                    except Exception as e:
                        print(f"Error in prediction: {e}")
                        print(f"unique_id: {id}")

                with open(f'llava16_temp.json', "w") as f:
                    json.dump(results, f, indent=4, default=str)
                    
                pbar.update(1)

    return results

# Main function
def main(dataset_name, split, processor, batch_size=32, save_path="llava-results.json", device="cuda"):
    # Load dataset
    dataset = load_hf_dataset(dataset_name, split)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load the model
    model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.to(device)

    # Perform evaluation
    scores = evaluate(model, dataloader, processor, device)

    # Save results
    with open(save_path, "w") as f:
        json.dump(scores, f, indent=4, default=str)

    print(f"Results saved to {save_path}.")

# Entry point
if __name__ == "__main__":
    time1 = time.time()

    default_output_path = os.path.join(os.getcwd(), "llava-results.json")
    parser = ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--save_path", type=str, default=default_output_path, help="Path to save the results")
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")

    main(args.dataset_name, args.split, processor, args.batch_size, args.save_path, device)
    print(f"Time taken: {time.time() - time1} seconds")

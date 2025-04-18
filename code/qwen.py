import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Features, Value, Image as HFImage
import json
from qwen_vl_utils import process_vision_info
import time
from argparse import ArgumentParser
from tqdm import tqdm
import os

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

# Load existing results
def load_existing_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            results = json.load(f)
        processed_ids = {result["id"] for result in results}
        return results, processed_ids
    return [], set()

# Evaluate function
def evaluate(model, dataloader, processor, device, temp_save_path="Qwen2VL_temp.json"):
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
                            "Analyze the following article and its associated image for signs of disinformation. "
                            "Respond with 'Classification: Likely' if the content is misleading, biased, or negative. "
                            "Respond with 'Classification: Unlikely' if the content appears accurate, credible, and unbiased. "
                            "Provide only one of the two classifications.\n\n"
                            f"Article: {text}\n\n"
                        )
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        
                        # Process inputs for the model
                        text = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to(device)
                        generated_ids = model.generate(
                            **inputs, max_new_tokens=512, temperature=0.7, top_p=0.9
                        )
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        predicted_answer = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # Validate the output
                        if "Likely" in predicted_answer[0]:
                            predicted_answer = "Classification: Likely"
                        elif "Unlikely" in predicted_answer[0]:
                            predicted_answer = "Classification: Unlikely"
                        else:
                            predicted_answer = "Unclear"

                        results.append({
                            "id": id,
                            "predicted_answer": predicted_answer,
                            "ground_truth": label,
                            "prompt": prompt
                        })

                    except Exception as e:
                        print(f"Error in prediction: {e}")
                        print(f"unique_id: {id}")

                # Save intermediate results
                with open(temp_save_path, "w") as f:
                    json.dump(results, f, indent=4, default=str)

                pbar.update(1)

    return results

# Main evaluation function
def main(dataset_name, split, batch_size=32, save_path="qwen_results.json", temp_save_path="Qwen2VL_temp.json", device="cuda"):
    # Load existing results
    existing_results, processed_ids = load_existing_results(temp_save_path)

    # Load dataset and filter out already processed examples
    dataset = load_hf_dataset(dataset_name, split)
    dataset = [item for item in dataset if item["unique_id"] not in processed_ids]

    if not dataset:
        print("All data has already been processed. Exiting.")
        with open(save_path, "w") as f:
            json.dump(existing_results, f, indent=4, default=str)
        return

    # Log remaining data
    print(f"Remaining items to process: {len(dataset)}")

    # Check label distribution for debugging
    label_distribution = {label: sum(1 for item in dataset if item['multimodal_label'] == label) for label in ["Likely", "Unlikely"]}
    print(f"Label Distribution: {label_distribution}")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Evaluate the model
    new_results = evaluate(model, dataloader, processor, device, temp_save_path)

    # Combine existing and new results
    final_results = existing_results + new_results

    # Save final results
    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=4, default=str)

    print(f"Results saved to {save_path}.")

if __name__ == "__main__":
    time1 = time.time()

    default_output_path = os.path.join(os.getcwd(), "qwen_results.json")
    temp_output_path = os.path.join(os.getcwd(), "Qwen2VL_temp.json")

    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--save_path", type=str, default=default_output_path, help="Path to save the results")
    parser.add_argument("--temp_save_path", type=str, default=temp_output_path, help="Path to save intermediate results")
    args = parser.parse_args()

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load the model
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).eval()

    # Run main evaluation
    main(args.dataset_name, args.split, args.batch_size, args.save_path, args.temp_save_path, args.device)
    print(f"Time taken: {time.time() - time1} seconds")

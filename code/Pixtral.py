import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value, Image as HFImage
import json
import time
from argparse import ArgumentParser
from tqdm import tqdm
import os
import base64
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login
from dotenv import load_dotenv


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
    # dataset = load_dataset(dataset_name, split=split, features=features)
    dataset = load_dataset(dataset_name, split=split)
    return dataset

# Function to truncate content to 1000 words
def truncate_to_2000_words(text):
    # Remove extra spaces and new lines
    cleaned_text = ' '.join(text.split())
    words = cleaned_text.split()
    return ' '.join(words[:2000]) if len(words) > 2000 else cleaned_text

# Collate function
def collate_fn(batch):
    images = [item['image'] for item in batch]  # Images
    article_texts = [item['article_text'] for item in batch]
    labels = [item['multimodal_label'] for item in batch]
    ids = [item['unique_id'] for item in batch]

    return {
        'images': images,
        'article_texts': article_texts,
        'answers': labels,
        'ids': ids
    }

def huggingface_image_to_data_url(hf_image, format: str = "JPEG") -> str:
    """
    Convert a Hugging Face image file to a data URL.
    
    Args:
        hf_image: Hugging Face image file object (e.g., a PIL Image).
        format (str): Desired image format (default is "JPEG").
    
    Returns:
        str: Data URL of the image.
    """
    # Save the image to an in-memory buffer
    from io import BytesIO
    buffer = BytesIO()
    hf_image.save(buffer, format=format)
    buffer.seek(0)  # Rewind to the beginning of the buffer

    # Encode the image data as Base64
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Determine MIME type based on format
    mime_type = f"image/{format.lower()}"

    # Construct and return the data URL
    return f"data:{mime_type};base64,{encoded_string}"


# Evaluate function
def evaluate(model, dataloader, processor, device):
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

                        prompt = "Assess the text and image below for potential disinformation (try finding Deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.\n" \
                        "If you find any of the listed rhetorical  techniques, then the article is likely disinformation; if not, it is likely not disinformation.\n\n" \
                        "Rhetorical Techniques Checklist:\n" \
                        "- Emotional Appeal: Uses language that intentionally invokes extreme emotions like fear or anger, aiming to distract from lack of factual backing.\n" \
                        "- Exaggeration and Hyperbole: Makes claims that are unsupported by evidence, or presents normal situations as extraordinary to manipulate perceptions.\n" \
                        "- Bias and Subjectivity: Presents information in a way that unreasonably favors one perspective, omitting key facts that might provide balance.\n" \
                        "- Repetition: Uses repeated messaging of specific points or misleading statements to embed a biased viewpoint in the reader's mind.\n" \
                        "- Specific Word Choices: Employs emotionally charged or misleading terms to sway opinions subtly, often in a manipulative manner.\n" \
                        "- Appeals to Authority: References authorities who lack relevant expertise or cites sources that do not have the credentials to be considered authoritative in the context.\n" \
                        "- Lack of Verifiable Sources: Relies on sources that either cannot be verified or do not exist, suggesting a fabrication of information.\n" \
                        "- Logical Fallacies: Engages in flawed reasoning such as circular reasoning, strawman arguments, or ad hominem attacks that undermine logical debate.\n" \
                        "- Conspiracy Theories: Propagates theories that lack proof and often contain elements of paranoia or implausible scenarios as facts.\n" \
                        "- Inconsistencies and Factual Errors: Contains multiple contradictions or factual inaccuracies that are easily disprovable, indicating a lack of concern for truth.\n" \
                        "- Selective Omission: Deliberately leaves out crucial information that is essential for a fair understanding of the topic, skewing perception.\n" \
                        "- Manipulative Framing: Frames issues in a way that leaves out alternative perspectives or possible explanations, focusing only on aspects that support a biased narrative.\n\n" \
                        f"Article: {text}\n\n" \
                        "Please provide your answer in the format: 'Likely' or 'Unlikely' and do not provide any explaination or text." 
                        
                        image = huggingface_image_to_data_url(img)
                        messages = [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image}}]
                            },
                        ]

                        outputs = llm.chat(messages, sampling_params=sampling_params)

                        predicted_answer = outputs[0].outputs[0].text
                        
                        results.append({
                            "id": id,
                            "predicted_answer": predicted_answer,
                            "ground_truth": label,
                            "prompt": prompt
                        })

                    except Exception as e:
                        print(f"Error in prediction: {e}")
                        print(f"unique_id: {id}")

                with open(f'/scripts/outputs/Pixtral_temp.json', "w") as f:
                    json.dump(results, f, indent=4, default=str)

                pbar.update(1)

    return results

# Main evaluation function
def main(dataset_name, split, processor, batch_size=32, save_path="results.json", device="cuda"):
    all_results = {}
    dataset = load_hf_dataset(dataset_name, split)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Perform evaluation
    scores = evaluate(None, dataloader, processor, device)

    with open(save_path, "w") as f:
        json.dump(scores, f, indent=4, default=str)

    print(f"Results saved to {save_path}.")



if __name__ == "__main__":
    time1 = time.time()

    default_output_path = os.path.join(os.getcwd(), "results.json")
    # add args
    parser = ArgumentParser()
    # device
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    # dataset
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Dataset name")
    # split
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate on")
    # batch size
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    # save_path
    parser.add_argument("--save_path", type=str, default=default_output_path, help="Path to save the results")

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the token from the environment variable
    token = os.getenv("--")

    # login(token=token)

    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load the model and tokenizer
    model_name = "mistralai/Pixtral-12B-2409"

    sampling_params = SamplingParams(max_tokens=8192)

    llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=32768, device=device)

    dataset_name = args.dataset_name
    split = args.split
    main(dataset_name, split,None, args.batch_size, args.save_path, args.device)
    print(f"Time taken: {time.time() - time1} seconds")

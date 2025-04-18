import os
import argparse
import pandas as pd
import random
import re
from textattack.augmentation import WordNetAugmenter, CharSwapAugmenter

# Define augmenters
AUGMENTERS = {
    "synonym": WordNetAugmenter(
        pct_words_to_swap=0.3,
        transformations_per_example=1
    ),
    "misspelling": CharSwapAugmenter(
        pct_words_to_swap=0.2,
        transformations_per_example=1
    )
}

# List of auxiliary/modal verbs for negation
AUXILIARIES = [
    r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b",
    r"\bcan\b", r"\bcould\b", r"\bshall\b", r"\bshould\b",
    r"\bwill\b", r"\bwould\b", r"\bmay\b", r"\bmight\b",
    r"\bdo\b", r"\bdoes\b", r"\bdid\b", r"\bhas\b", r"\bhave\b", r"\bhad\b"
]

def insert_negation(text):
    # Try inserting "not" or "never" after a modal/aux verb
    for aux in AUXILIARIES:
        match = re.search(aux, text, re.IGNORECASE)
        if match:
            insert_word = random.choice(["not", "never"])
            start = match.end()
            return text[:start] + f" {insert_word}" + text[start:]
    return text  # fallback: no change

def apply_perturbation(input_csv, output_csv, perturbation_type):
    df = pd.read_csv(input_csv)

    if 'ID' not in df.columns or 'Article_text' not in df.columns:
        raise ValueError("CSV must contain 'ID' and 'Article_text' columns.")

    def controlled_augment(sentence):
        if perturbation_type == "negation":
            return insert_negation(sentence)
        else:
            augmenter = AUGMENTERS[perturbation_type]
            augmented = augmenter.augment(sentence)
            if not augmented:
                return sentence
            if perturbation_type == "synonym":
                # Keep only 1–2 word swaps
                original_words = sentence.split()
                changed = [
                    aug for aug in augmented
                    if 1 <= sum(o != a for o, a in zip(original_words, aug.split())) <= 2
                ]
                return changed[0] if changed else sentence
            else:
                return augmented[0]

    df["Augmented_text"] = df["Article_text"].apply(controlled_augment)
    df.to_csv(output_csv, index=False)
    print(f"Saved augmented data with '{perturbation_type}' perturbation to {output_csv}")

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply text perturbations using TextAttack and custom rules.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with 'ID' and 'Article_text'.")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV.")
    parser.add_argument("--perturbation", default="synonym", choices=["synonym", "misspelling", "negation"],
                        help="Type of perturbation to apply (default: synonym).")
    args = parser.parse_args()

    apply_perturbation(args.input_csv, args.output_csv, args.perturbation)



# To Run:
# python text_perturbations.py --input_csv path/to/input.csv --output_csv path/to/output.csv --perturbation synonym
# python text_perturbations.py --input_csv path/to/input.csv --output_csv path/to/output.csv --perturbation misspelling
# python text_perturbations.py --input_csv path/to/input.csv --output_csv path/to/output.csv --perturbation negation

# To Run all perturbations:
# python text_perturbations.py --input_csv path/to/input.csv --output_csv path/to/output.csv --perturbation all
import pandas as pd
from PIL import Image
from IPython.display import display
import os

# Load the Parquet file
metadata_path_parquet = "data/metadata.parquet"
df_metadata = pd.read_parquet(metadata_path_parquet)

# Display the first 5 rows of the DataFrame
print("First 5 rows of the metadata:")
print(df_metadata.head(5))

# Check and display images
for i, row in df_metadata.head(5).iterrows():
    image_path = row.get('saved_image_path')  # Adjust the column name if necessary
    print(f"Checking image for entry {i}: {image_path}")
    if image_path and os.path.exists(image_path):  # Check if the file exists
        try:
            img = Image.open(image_path)
            display(img)  # Display the image in Jupyter Notebook
            print(f"Image {i} loaded successfully.")
        except Exception as e:
            print(f"Error displaying image {i}: {e}")
    else:
        print(f"Image path is invalid or file does not exist for entry {i}.")

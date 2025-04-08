import base64
import tempfile
import zipfile
import csv
import os
from openai import OpenAI, RateLimitError, APIError
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Initialize client (relies on OPENAI_API_KEY env var by default)
try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Ensure OPENAI_API_KEY environment variable is set.")
    exit()

# Function to encode the image (keep as is)
def encode_image(image_path):
    try:
        # Optional: Add a check to ensure it's a valid image first
        img = Image.open(image_path)
        img.verify() # Verify image header is okay
        # Re-open after verify
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
        print(f"Warning: Cannot encode image {image_path}. Error: {e}")
        return None

# --- Revised get_caption ---
def get_caption(base64_image: str, custom_suffix: str = "") -> str:
    """Gets caption from GPT-4 Vision using the OpenAI client."""
    if not base64_image:
        return "Invalid image provided"

    # Base prompt - Keep your concise prompt
    custom_prompt = "Directly describe with brevity and as brief as possible the scene or characters without any introductory phrase like 'This image shows', 'In the scene', 'This image depicts' or similar phrases. Just start describing the scene please. Do not end the caption with a '.'. Some characters may be animated, refer to them as regular humans and not animated humans. Please make no reference to any particular style or characters from any TV show or Movie. Good examples: a cat on a windowsill, a photo of smiling cactus in an office, a man and baby sitting by a window, a photo of wheel on a car"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100 # Reduced tokens slightly, often enough for brief captions
        )

        if response.choices:
            caption = response.choices[0].message.content.strip()
            # Remove potential problematic characters if strictly needed, but reconsider
            # caption = caption.replace('"', '') # Maybe keep comma?
            # Append the custom suffix IF it was provided
            if custom_suffix:
                return f"{caption} {custom_suffix}"
            else:
                return caption
        else:
            return "Failed to get caption (no choices)"

    except RateLimitError:
        print("Error: OpenAI API rate limit exceeded. Please wait and try again.")
        return "Failed to get caption (Rate Limit)"
    except APIError as e:
        print(f"Error: OpenAI API error: {e}")
        return f"Failed to get caption (API Error: {e.status_code})"
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        return "Failed to get caption (Unexpected Error)"


# --- Revised process_images ---
def process_images(input_path: str, output_csv: str, custom_suffix: str):
    """Processes images from a directory or zip file and writes captions to CSV."""
    images_to_process = []
    base_dir = input_path # Assume it's a directory initially

    # Handle zip file extraction
    if zipfile.is_zipfile(input_path):
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        print(f"Extracting zip file to temporary directory: {temp_dir}")
        try:
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            base_dir = temp_dir # Process files from the temp dir
        except zipfile.BadZipFile:
            print(f"Error: Invalid zip file provided at {input_path}")
            temp_dir_obj.cleanup()
            return
        except Exception as e:
            print(f"Error extracting zip file: {e}")
            temp_dir_obj.cleanup()
            return
    elif not os.path.isdir(input_path):
        print(f"Error: Input path {input_path} is not a valid directory or zip file.")
        return

    # Collect all image file paths first
    print(f"Scanning for images in: {base_dir}")
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                images_to_process.append(os.path.join(root, file_name))

    print(f"Found {len(images_to_process)} image files.")
    if not images_to_process:
        print("No images found to process.")
        if 'temp_dir_obj' in locals(): temp_dir_obj.cleanup() # Clean up temp dir if created
        return

    # Process images and write to CSV
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['image_path', 'caption'])  # Header row (use image_path)

            for image_path in tqdm(images_to_process, desc="Captioning Images"):
                base64_image = encode_image(image_path)
                if base64_image: # Only proceed if encoding was successful
                    caption = get_caption(base64_image, custom_suffix)
                    # Use relative path for portability
                    relative_path = os.path.relpath(image_path, start=base_dir)
                    # print(f"Image: {relative_path}, Caption: {caption}\n") # Optional print
                    writer.writerow([relative_path, caption])
                else:
                    # Write error or skip? Optional: write placeholder
                    relative_path = os.path.relpath(image_path, start=base_dir)
                    writer.writerow([relative_path, "Error encoding image"])

    except IOError as e:
        print(f"Error writing to CSV file {output_csv}: {e}")
    finally:
        # Cleanup temp directory if it was created
        if 'temp_dir_obj' in locals():
            print(f"Cleaning up temporary directory: {temp_dir}")
            temp_dir_obj.cleanup()

# --- Revised main ---
def main():
    input_path = input("Enter the path to the image folder or zip file: ")
    output_csv = input("Enter the desired output CSV file name (e.g., captions.csv): ") or "captions.csv"

    # Simplified suffix logic
    add_suffix = input("Add a custom suffix like 'in the style of X' or 'as Y'? (y/n): ").strip().lower()
    custom_suffix = ""
    if add_suffix == 'y':
        custom_suffix = input("Enter the exact suffix text (e.g., 'in the style of TOK', 'as a Family Guy character'): ").strip()

    # API key check (client initialization already attempts this)
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        return # Exit if key definitely not set

    print("\nStarting processing...")
    process_images(input_path, output_csv, custom_suffix)
    print(f"\nProcessing complete. Captions saved to {output_csv}")
    print("Note: GPT-4 Vision API calls can be slow and incur costs.")

if __name__ == "__main__":
    main()
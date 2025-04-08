import os
import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm  # For progress bar
import time
import argparse
from typing import List, Tuple
import csv

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions for a dataset of images using BLIP model")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to directory containing images")
    parser.add_argument("--output_file", type=str, default="image_captions.csv",
                        help="Path to output CSV file (default: image_captions.csv)")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip-image-captioning-large",
                        help="Model ID from Hugging Face (default: Salesforce/blip-image-captioning-large)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing (default: 8, use smaller value if low on memory)")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated captions (default: 50)")
    parser.add_argument("--use_relative_paths", action="store_true",
                        help="Store relative paths in the output file instead of absolute paths")
    parser.add_argument("--delimiter", type=str, default=",", choices=[",", "\t"],
                        help="Delimiter for output file (default: comma)")
    return parser.parse_args()

# --- Helper Functions ---
def find_image_files(directory: str) -> List[str]:
    """Recursively finds all image files (jpg, jpeg, png) in a directory."""
    image_files = []
    print(f"Scanning for images in: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    print(f"Found {len(image_files)} image files.")
    return image_files

def process_batch(model, processor, image_paths: List[str], device: str, max_length: int) -> List[Tuple[str, str]]:
    """Process a batch of images and return their captions."""
    results = []
    batch_inputs = []
    valid_paths = []
    
    # Prepare batch
    for img_path in image_paths:
        try:
            raw_image = Image.open(img_path).convert('RGB')
            inputs = processor(images=raw_image, return_tensors="pt")
            batch_inputs.append(inputs)
            valid_paths.append(img_path)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Warning: Error loading image {img_path}: {e}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    # If no valid images found in this batch
    if not batch_inputs:
        return []
    
    # Create batch dictionary
    if len(batch_inputs) == 1:
        # Single image case
        batch_dict = batch_inputs[0].to(device)
    else:
        # Multiple images case
        batch_dict = {
            "pixel_values": torch.cat([x["pixel_values"] for x in batch_inputs]).to(device)
        }
    
    # Generate captions
    try:
        with torch.no_grad():
            generated_ids = model.generate(**batch_dict, max_length=max_length)
            
        # Decode captions
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Create result pairs (path, caption)
        results = list(zip(valid_paths, captions))
    except Exception as e:
        print(f"Error during batch generation: {e}")
    
    return results

# --- Main Processing ---
def generate_captions(args):
    start_time = time.time()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu" and args.batch_size > 1:
        print("WARNING: Running on CPU with batch size > 1 may be slow. Consider setting batch_size=1.")
    
    # 1. Load Model and Processor
    print(f"Loading model: {args.model_id}...")
    try:
        processor = BlipProcessor.from_pretrained(args.model_id)
        model = BlipForConditionalGeneration.from_pretrained(args.model_id).to(device)
        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have installed 'transformers' and 'torch',")
        print("and that the model ID is correct and you have internet access.")
        return

    # 2. Find Image Files
    image_paths = find_image_files(args.dataset_dir)
    if not image_paths:
        print("No image files found. Please check the dataset directory path.")
        return

    total_images = len(image_paths)
    print(f"Starting caption generation for {total_images} images with batch size {args.batch_size}...")

    # 3. Generate Captions and Write to File
    captions_generated = 0
    error_count = 0
    
    # File extension check
    file_ext = os.path.splitext(args.output_file)[1].lower()
    if args.delimiter == "," and file_ext != ".csv":
        print(f"Warning: Using comma delimiter but file extension is not .csv: {args.output_file}")
    elif args.delimiter == "\t" and file_ext != ".tsv":
        print(f"Warning: Using tab delimiter but file extension is not .tsv: {args.output_file}")
        
    try:
        with open(args.output_file, 'w', encoding='utf-8', newline='') as f_out:
            # Create CSV writer
            writer = csv.writer(f_out, delimiter=args.delimiter)
            
            # Write header
            writer.writerow(["image_path", "caption"])

            # Process in batches
            for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Processing Batches"):
                batch_paths = image_paths[i:i+args.batch_size]
                results = process_batch(model, processor, batch_paths, device, args.max_length)
                
                # Write results
                for img_path, caption in results:
                    # Use relative path if specified
                    if args.use_relative_paths:
                        path_to_write = os.path.relpath(img_path, start=args.dataset_dir)
                    else:
                        path_to_write = img_path
                        
                    writer.writerow([path_to_write, caption.strip()])
                    captions_generated += 1

    except IOError as e:
        print(f"Error opening or writing to output file {args.output_file}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print("\n--- Processing Complete ---")
        print(f"Successfully generated captions for: {captions_generated} images")
        print(f"Encountered errors on: {total_images - captions_generated} images")
        print(f"Captions saved to: {args.output_file}")
        print(f"Total processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        if device == "cpu":
            print("Consider using a GPU for significantly faster processing next time.")

# --- Run the process ---
if __name__ == "__main__":
    args = parse_args()
    
    # Basic check for dataset directory
    if not os.path.isdir(args.dataset_dir):
        print("="*50)
        print(f"ERROR: Dataset directory not found: {args.dataset_dir}")
        print("Please provide a valid directory path with the --dataset_dir argument.")
        print("="*50)
    else:
        generate_captions(args)
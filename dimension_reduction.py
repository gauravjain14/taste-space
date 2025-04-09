import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
import os
import warnings
import argparse


# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str,
        default="shhq_samples_system.csv", help="Path to the CSV file containing captions")
    parser.add_argument("--image_dir", type=str,
        default="SHHQ-1.0_samples", help="Path to the directory containing images")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.05, help="UMAP parameter")
    parser.add_argument("--umap_n_components", type=int, default=2, help="UMAP parameter")
    parser.add_argument("--output_plot_filename", type=str, default="taste_space_visualization.png", help="Output plot filename")
    parser.add_argument("--plot_thumbnail_zoom", type=float, default=0.5, help="Thumbnail zoom factor")
    
    return parser.parse_args()


def main(args):
    print(f"Loading data from {args.csv_path}...")
    try:
        df = pd.read_csv(args.csv_path, header=None, names=['filename', 'caption'])
        print(f"Loaded {len(df)} entries.")
        # Optional: Check if IMAGE_DIR exists
        if not os.path.isdir(args.image_dir):
            print(f"Error: Image directory '{args.image_dir}' not found.")
            exit()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        exit()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit()

    # Load the CLIP model
    processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    model = CLIPModel.from_pretrained(args.clip_model_name).to(DEVICE)

    # Store embeddings and valid image paths (for visualization later)
    all_embeddings = []
    valid_image_paths = []
    filenames_for_plot = []

    print("Generating embeddings (this might take a while)...")
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing items"):
            filename = row['filename'].strip()
            caption = row['caption'].strip()
            image_path = os.path.join(args.image_dir, filename)

            try:
                # Load and process image
                image = Image.open(image_path).convert("RGB")
                image_input = processor(images=image, return_tensors="pt", padding=True).to(DEVICE)
                image_embedding = model.get_image_features(**image_input)

                # Process text
                text_input = processor(text=caption, return_tensors="pt", padding=True).to(DEVICE)
                text_embedding = model.get_text_features(**text_input)

                # --- Average Embeddings ---
                # Normalize embeddings before averaging (good practice)
                image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                combined_embedding = (image_embedding_norm + text_embedding_norm) / 2.0

                # Store results (move embedding to CPU and convert to numpy)
                all_embeddings.append(combined_embedding.squeeze().cpu().numpy())
                valid_image_paths.append(image_path) # Keep track of images we successfully processed
                filenames_for_plot.append(filename)

            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: Error processing {filename}: {e}. Skipping.")
                continue

    if not all_embeddings:
        print("Error: No embeddings were generated. Check image paths and file integrity.")
        exit()

    all_embeddings_np = np.array(all_embeddings)
    print(f"Generated {len(all_embeddings_np)} combined embeddings.")

    # Perform dimensionality reduction
    print("Running UMAP for dimensionality reduction...")
    try:
        reducer = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            n_components=args.umap_n_components,
            metric='cosine', # Cosine distance is often good for normalized embeddings
            random_state=42 # for reproducibility
        )
        embeddings_2d = reducer.fit_transform(all_embeddings_np)
        print("UMAP reduction complete.")
        print(f"Shape of 2D embeddings: {embeddings_2d.shape}")

    except Exception as e:
        print(f"Error during UMAP reduction: {e}")
        exit()

    print("Creating visualization...")
    def plot_image_scatter(images, coords, zoom=0.1, figsize=(20, 20)):
        """ Plots images at specified 2D coordinates. """
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0) # Plot invisible points just to set axes limits

        artists = []
        for image_path, (x, y) in tqdm(zip(images, coords),
                                      total=len(images), desc="Plotting images"):
            try:
                img = Image.open(image_path).convert("RGB")
                img.thumbnail((100, 100)) # Resize thumbnail for performance
                im = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
                artists.append(ax.add_artist(ab))
            except Exception as e:
                print(f"Warning: Could not plot image {os.path.basename(image_path)}: {e}")

        ax.update_datalim(coords)
        ax.autoscale()
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title("2D Taste Space Visualization (CLIP Embeddings + UMAP)")
        # Optional: Hide axes ticks for a cleaner look like the example
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.axis('off')

        plt.tight_layout()
        return fig, ax

    # Create the plot
    fig, ax = plot_image_scatter(valid_image_paths, embeddings_2d, zoom=args.plot_thumbnail_zoom)

    # Save the plot
    try:
        plt.savefig(args.output_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.output_plot_filename}")
        plt.show() # Also display the plot
    except Exception as e:
        print(f"Error saving plot: {e}")

    print("Script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from model.model import load_model, smiles_to_ecfp4

def extract_training_data(
    model_path="model/model.safetensors",
    data_path="zinc_250k.txt",
    output_path="reconstructed_training_data.csv",
    batch_size=1000,
    top_k=1000
):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    model.eval()

    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        # Ignore empty lines and header if any
        smiles_list = [line.strip() for line in f.readlines() if line.strip() and line.strip() != 'smiles']
    
    total_candidates = len(smiles_list)
    print(f"Total candidates to evaluate: {total_candidates}")

    all_scores = []
    valid_smiles = []

    print("Evaluating dataset...")
    # Use tqdm if installed, otherwise simple loop
    for i in range(0, total_candidates, batch_size):
        batch_smiles = smiles_list[i : i + batch_size]

        # Convert SMILES to ECFP4 fingerprints
        ecfp_list = [smiles_to_ecfp4(s) for s in batch_smiles]
        
        # Filter out invalid molecules
        valid_indices = [idx for idx, fp in enumerate(ecfp_list) if fp is not None]
        valid_fps = [ecfp_list[idx] for idx in valid_indices]
        
        if not valid_fps:
            continue

        X = torch.from_numpy(np.asarray(valid_fps, dtype=np.float32))

        with torch.no_grad():
            logits = model(X) # (N, 2)
            
            # Calculate the absolute difference between logits (Confidence Score)
            # Higher absolute difference means the model is highly confident (memorized)
            abs_logit_diff = torch.abs(logits[:, 1] - logits[:, 0]).cpu().numpy()
            
        # Store valid SMILES and their scores
        for idx, score in zip(valid_indices, abs_logit_diff):
            valid_smiles.append(batch_smiles[idx])
            all_scores.append(score)

        if i % (batch_size * 10) == 0 and i > 0:
            print(f"Processed {i}/{total_candidates} molecules...")

    print("Evaluation complete. Sorting results...")
    
    # Create a DataFrame for easy sorting and saving
    df = pd.DataFrame({
        'smiles': valid_smiles,
        'confidence_score': all_scores
    })

    # Sort descending by confidence score
    df_sorted = df.sort_values(by='confidence_score', ascending=False).reset_index(drop=True)

    # Save the top K results to a CSV file
    top_results = df_sorted.head(top_k)
    top_results.to_csv(output_path, index=False)
    
    print(f"\nTop 5 extracted molecules:")
    print(top_results.head(5))
    
    print(f"\nSuccessfully extracted and saved the top {top_k} reconstructed molecules to {output_path}.")

if __name__ == "__main__":
    # Ensure paths are correct based on the directory structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, "model", "model.safetensors")
    data_file = os.path.join(current_dir, "zinc_250k.txt")
    out_file = os.path.join(current_dir, "reconstructed_training_data.csv")
    
    extract_training_data(
        model_path=model_file,
        data_path=data_file,
        output_path=out_file,
        batch_size=2000,
        top_k=1000 # Number of molecules to extract as the reconstructed training set
    )

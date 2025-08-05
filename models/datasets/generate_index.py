
import os
import json
from safetensors.torch import load_file

def generate_safetensors_index(weights_dir, output_index_path):
    """
    Generates a model.safetensors.index.json file for Hugging Face format.

    - Scans all .safetensors files in the weights_dir.
    - Maps each tensor name to its corresponding .safetensors file.
    - Saves the index as a JSON file.

    Args:
        weights_dir (str): Directory containing .safetensors files.
        output_index_path (str): Path to save the model.safetensors.index.json file.
    """
    if not os.path.isdir(weights_dir):
        print(f"Error: Directory '{weights_dir}' not found.")
        return

    weight_map = {}
    safetensors_files = [f for f in os.listdir(weights_dir) if f.endswith(".safetensors")]

    if not safetensors_files:
        print(f"No .safetensors files found in {weights_dir}.")
        return

    print(f"Found {len(safetensors_files)} .safetensors files in {weights_dir}.")

    for filename in safetensors_files:
        file_path = os.path.join(weights_dir, filename)
        print(f"Processing file: {file_path}")

        try:
            # Load tensors from the safetensors file
            tensors = load_file(file_path, device='cpu')
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            continue

        # Map each tensor name to the safetensors file
        for tensor_name in tensors.keys():
            weight_map[tensor_name] = filename
            # print(f"  Mapped tensor '{tensor_name}' to '{filename}'")

    if not weight_map:
        print("No tensors found in any .safetensors files. Index file not generated.")
        return

    # Create the index dictionary
    index_data = {
        "metadata": {
            "total_size": sum(
                os.path.getsize(os.path.join(weights_dir, f))
                for f in safetensors_files
            )
        },
        "weight_map": weight_map
    }

    # Save the index file
    try:
        with open(output_index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        print(f"Successfully generated index file: {output_index_path}")
        print(f"Total tensors mapped: {len(weight_map)}")
    except Exception as e:
        print(f"Error saving index file {output_index_path}: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    weights_directory = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/compressed/3bitvq_nohessian/constructed_model_HF_format"
    output_index_file = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/compressed/3bitvq_nohessian/constructed_model_HF_format/model.safetensors.index.json"
    # --- End Configuration ---

    generate_safetensors_index(weights_directory, output_index_file)

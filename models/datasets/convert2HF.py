import os
import torch
from safetensors.torch import load_file, save_file

def convert_safetensors_to_hf_format(source_dir, dest_dir):
    """
    Converts custom safetensors files to Hugging Face standard format.

    - Removes tensors with 'assignments', 'codebook', 'normalizer' in their names.
    - Renames 'cached_reconstruct' to 'weight' in tensor names.
    - Keeps all other tensors unchanged.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")

    processed_files = 0
    for filename in os.listdir(source_dir):
        if filename.endswith(".safetensors"):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)

            print(f"\nProcessing file: {source_path}")

            try:
                # Load tensors from the source file
                tensors = load_file(source_path, device='cpu')
            except Exception as e:
                print(f"  Error loading {source_path}: {e}")
                continue

            hf_tensors = {}
            original_tensor_count = len(tensors)
            kept_tensor_count = 0
            skipped_tensor_count = 0

            for name, tensor_data in tensors.items():
                # Check for keywords to discard
                if "assignments" in name or \
                   "codebook" in name or \
                   "normalizer" in name:
                    # print(f"  Skipping (keyword): {name}")
                    skipped_tensor_count += 1
                    continue

                # Check for the specific tensor to rename
                if "cached_reconstruct" in name:
                    new_name = name.replace("cached_reconstruct", "weight")
                    hf_tensors[new_name] = tensor_data
                    # print(f"  Renaming: '{name}' -> '{new_name}' (Shape: {tensor_data.shape}, Dtype: {tensor_data.dtype})")
                    kept_tensor_count += 1
                else:
                    # Keep other tensors unchanged
                    hf_tensors[name] = tensor_data
                    # print(f"  Keeping unchanged: '{name}' (Shape: {tensor_data.shape}, Dtype: {tensor_data.dtype})")
                    kept_tensor_count += 1

            if hf_tensors:
                try:
                    save_file(hf_tensors, dest_path)
                    print(f"  Successfully saved {kept_tensor_count} tensors to {dest_path}.")
                    print(f"  Original tensors: {original_tensor_count}, Kept: {kept_tensor_count}, Skipped: {skipped_tensor_count}")
                    processed_files += 1
                except Exception as e:
                    print(f"  Error saving {dest_path}: {e}")
            else:
                print(f"  No tensors to save for {dest_path} after filtering.")
                print(f"  Original tensors: {original_tensor_count}, Kept: 0, Skipped: {original_tensor_count}")

    print(f"\n--- Conversion Complete ---")
    print(f"Total files processed and saved: {processed_files}")
    print(f"Processed files are in: {dest_dir}")

if __name__ == "__main__":
    # --- Configuration ---
    source_directory = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/compressed/3bitvq_nohessian/constructed_model"
    destination_directory = "/data01/home/zhaozx/NoWag/models/datasets/Qwen3-30B-A3B/compressed/3bitvq_nohessian/constructed_model_HF_format"  # 替换为你想要的路径
    # --- End Configuration ---

    if not os.path.isdir(source_directory):
        print(f"Error: Source directory '{source_directory}' not found.")
    else:
        convert_safetensors_to_hf_format(source_directory, destination_directory)
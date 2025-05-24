import os
import soundfile as sf
from datasets import load_from_disk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = "/mnt/scratch/pippalin2/jupyter/GMM-DistilHuBERT"
DATASET_PATH = os.path.join(REPO_ROOT, "data/hf_librispeech_clean100")
OUTPUT_DIR = os.path.join(REPO_ROOT, "mfa/librispeech_wavtxt")

# Global dataset to be accessible inside each process
dataset = None

def init_worker():
    global dataset
    dataset = load_from_disk(DATASET_PATH)

def export_example(index):
    global dataset
    try:
        example = dataset[index]
        base_name = f"{index:05d}"
        wav_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
        txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")

        if os.path.exists(wav_path) and os.path.exists(txt_path):
            return f"Skipped {index} (already exists)"

        sf.write(wav_path, example["audio"]["array"], example["audio"]["sampling_rate"])
        with open(txt_path, "w") as f:
            f.write(example["text"])
        return True

    except Exception as e:
        return f"Error at index {index}: {e}"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    temp = load_from_disk(DATASET_PATH)  # Just to check length
    print(f"Dataset loaded with {len(temp)} samples.")

    index_list = list(range(len(temp)))

    print("Starting parallel export...")
    with ProcessPoolExecutor(initializer=init_worker) as executor:
        futures = [executor.submit(export_example, i) for i in index_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Exporting"):
            result = future.result()
            if result is not True:
                print(result)

if __name__ == "__main__":
    main()

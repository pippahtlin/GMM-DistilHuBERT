#%%
import torchaudio

# This will download the subset you want (e.g., "train-clean-100") if not already cached
dataset = torchaudio.datasets.LIBRISPEECH(
    root="./data",  # where to store the data
    url="train-clean-100",  # or train-clean-360, train-other-500, train-960, etc.
    download=True
)

# Example: get first audio sample
waveform, sample_rate, transcript, _, _, _ = dataset[0]

# %%

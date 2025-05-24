import torch

def extract_all_layer_features(model, waveform, layers=[0, 1, 2]):
    with torch.no_grad():
        outputs = model(waveform, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: [0]=embedding, [1]=layer1, etc.
        selected = [hidden_states[i] for i in layers]
        return torch.cat(selected, dim=-1)  # [1, T, D_total]

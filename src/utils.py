import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def add_noise_to_embeddings(embeddings, noise_level=0.1):
    noise = torch.randn_like(embeddings) * noise_level
    return embeddings + noise


  
class GloveDataset(Dataset):
    def __init__(self, embeddings, sequence_length, batch_size):
        self.embeddings = embeddings
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Cosine Similarity for Accuracy
def compute_accuracy(predicted, target, threshold=0.5):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy

def add_noise_to_embeddings(embeddings, noise_level=0.1):
    noise = torch.randn_like(embeddings) * noise_level
    return embeddings + noise

# Load GloVe Embeddings Manually
def load_glove_embeddings(file_path, vocab_size=5000):
    """Load GloVe embeddings from a file for a small subset."""
    embeddings = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= vocab_size:
                break
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
            embeddings[word] = vector
    return embeddings

# Prepare Input Embeddings
def prepare_embeddings(glove_embeddings, batch_size, sequence_length, dim=300):
    """Randomly sample embeddings from the loaded GloVe vectors."""
    selected_vectors = torch.stack(
        [glove_embeddings[word] for word in list(glove_embeddings.keys())[:sequence_length]]
    )
    input_embeddings = selected_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
    return input_embeddings
  

  
class GloveDataset(Dataset):
    def __init__(self, embeddings, sequence_length, batch_size):
        self.embeddings = embeddings
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]
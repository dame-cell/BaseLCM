import torch 
import torch.nn.functional as F

def add_noise_to_embeddings(embeddings, noise_level=0.1):
    noise = torch.randn_like(embeddings) * noise_level
    return embeddings + noise

# Cosine Similarity for Accuracy
def compute_accuracy(predicted, target, threshold=0.5):
    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
    correct = (cos_sim > threshold).float()
    accuracy = correct.mean().item()
    return accuracy

class Dataset:
  pass 

class DataLoader:
  pass 
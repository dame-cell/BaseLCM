import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import spacy
from spacy.tokens import Doc
from tqdm.auto import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Iterator
import itertools

def setup_spacy():
    """Set up spaCy with optimal settings for batch processing"""
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    # Only enable the sentence segmenter
    nlp.enable_pipe('senter')
    return nlp

def process_batch(texts: List[str], nlp, batch_size: int = 1000) -> List[str]:
    """Process a batch of texts efficiently"""
    sentences = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        sentences.extend([sent.text.strip() for sent in doc.sents])
    return sentences

# Move process_chunk outside to make it picklable
def process_chunk(chunk: List[str], batch_size: int = 1000) -> List[str]:
    """Process a chunk of texts"""
    nlp = setup_spacy()
    return process_batch(chunk, nlp, batch_size)

def parallel_process_texts(texts: List[str], n_workers: int = 4, batch_size: int = 1000) -> List[str]:
    """Process texts in parallel using multiple workers"""
    # Split texts into roughly equal chunks for parallel processing
    chunk_size = max(len(texts) // n_workers, 1)  # Ensure chunk_size is at least 1
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    sentences = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a list of batch_size arguments for each chunk
        batch_sizes = [batch_size] * len(chunks)
        # Process chunks in parallel and show progress
        results = list(tqdm(
            executor.map(process_chunk, chunks, batch_sizes),
            total=len(chunks),
            desc="Processing text chunks"
        ))
        # Combine results from all workers
        sentences = list(itertools.chain(*results))
    
    return sentences
    

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

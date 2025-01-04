import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from tqdm.auto import tqdm
import torch

class SonarEncoder:
    def __init__(self, model_name='cointegrated/SONAR_200_text_encoder', device="cpu"):
        self.encoder = M2M100Encoder.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def encode(self, texts, lang, batch_size=32, norm=False):
        if self.tokenizer is None or self.encoder is None:
            raise ValueError("Tokenizer or encoder is not initialized.")

        self.tokenizer.src_lang = lang
        texts = texts if isinstance(texts, list) else [texts]

        embeddings = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Batches", unit="batch"):
                batch_texts = texts[i:i + batch_size]
                batch = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
                seq_embs = self.encoder(**batch).last_hidden_state
                mask = batch.attention_mask

                # Compute mean embedding for each sequence
                mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
                if norm:
                    mean_emb = torch.nn.functional.normalize(mean_emb, dim=1)

                embeddings.append(mean_emb)

        return torch.cat(embeddings, dim=0)

  
class PreNet(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.scaler_mean = 0.0 
        self.scaler_std = 1.0   

    def normalize(self, x):
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, x):
        x = self.normalize(x)
        x = self.linear(x)
        return x

class PostNet(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.scaler_mean = 0.0  
        self.scaler_std = 1.0   

    def denormalize(self, x):
        return x * self.scaler_std + self.scaler_mean

    def forward(self, x):
        x = self.linear(x)
        x = self.denormalize(x)
        return x
      
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, ff_dim, dropout=0.1, max_seq_len=512):
        super(TransformerDecoder, self).__init__()
        # Add mask to ensure causal attention
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True  
            )
            for _ in range(num_layers)
        ])
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
    def forward(self, x):
        seq_len = x.size(1)
        # Add positional encodings
        pos_enc = self.pos_encoder[:, :seq_len, :]
        x = x + pos_enc
        mask = self.causal_mask[:seq_len, :seq_len]
        for layer in self.layers:
            x = layer(x, x, tgt_mask=mask)
        return x
    
class BaseLCM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim):
        super(BaseLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.transformer_decoder = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim)
        self.postnet = PostNet(hidden_dim, output_dim)
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.prenet(x)
        x = self.transformer_decoder(x)
        x = self.postnet(x)
        return x.squeeze(1)  # Remove sequence dimension if single step
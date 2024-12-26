import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

# TO DO: Modify this to handle the texts correctly 
class SonarEncoder:
  """
  SONAR Encoder: Encodes sentences into embeddings using the SONAR model.
  """
  def __init__(self,model_name='cointegrated/SONAR_200_text_encoder',device="cpu"):
      self.encoder = M2M100Encoder.from_pretrained(model_name)
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.device = device
      
  def encode(self,texts,lang,norm=False):
    # this works for both single texts and batches of text
    if self.tokenizer is not None:
      self.tokenizer.src_lang = lang
      with torch.inference_mode():
        batch = self.tokenizer(texts, return_tensors='pt', padding=True)
        seq_embs = self.encoder(**batch).last_hidden_state
        mask = batch.attention_mask
        mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
        if norm:
            mean_emb = torch.nn.functional.normalize(mean_emb)
    return mean_emb
  
  
class PreNet(nn.Module):
  """
  PreNet : Maps the input features  to the model's hidden dimesntion after normalizing the input features.
  """
  def __init__(self,input_dim,hidden_dim):
    super(PreNet,self).__init__()
    self.linear = nn.Linear(input_dim,hidden_dim)
    self.scaler_mean = 0.0 
    self.scaler_std  = 1.0 
  
  def normalize(self, x):
        return (x - self.scaler_mean) / self.scaler_std

  def forward(self, x):
      x = self.normalize(x)
      x = self.linear(x)
      return x  


class PostNet(nn.Module):
    """
    Maps hidden state outputs back to the embedding space with denormalization.
    """
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
    def __init__(self,hidden_dim,num_heads,num_layers,ff_dim,dropout=0.1,max_seq_len=512):
        super(TransformerDecoder,self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
            )
            for _ in range(num_layers)
        ])
      
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

    def forward(self,x):
        seq_len = x.size(1)
        # Ensure we don't exceed the input sequence length
        pos_enc = self.pos_encoder[:, :seq_len, :]
        x = x + pos_enc
        for layer in self.layers:
            x = layer(x,x)
        return x
    
class BaseLCM(nn.Module):
  """
  Base Large Concept Model (LCM):
  - PreNet: Maps input embeddings to hidden space.
  - TransformerDecoder: Autoregressively processes embeddings.
  - PostNet: Maps output back to the embedding space.
  """
  def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_dim, output_dim):
      super(BaseLCM, self).__init__()
      self.prenet = PreNet(input_dim, hidden_dim)
      self.transformer_decoder = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim)
      self.postnet = PostNet(hidden_dim, output_dim)

  def forward(self, x):

    # Add sequence dimension if not present
    if len(x.shape) == 2:
        x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, input_dim]
    x = self.prenet(x)
    x = self.transformer_decoder(x)
    x = self.postnet(x)
    return x


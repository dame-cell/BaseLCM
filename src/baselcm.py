import torch
import torch.nn as nn
import torch.nn.functional as F


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
  """
  A normal transformer Decoder 
  """
  def __init__(self,hidden_dim,num_heads,num_layers,ff_dim,dropout):
    super(TransformerDecoder,self).__init__()
    self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
            )
            for _ in range(num_layers)
        ])
    self.pos_encoder = nn.Parameter(torch.zeros(1,512,hidden_dim))

  def forward(self,x):
    seq_len = x.size(1)
    x = x+ self.pos_encoder[:,:seq_len]
    for layer in self.layers:
      x = layer(x)
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
      x = self.prenet(x)
      x = self.transformer_decoder(x)
      x = self.postnet(x)
      return x




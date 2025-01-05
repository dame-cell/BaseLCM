import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb 
import argparse  
from baselcm import BaseLCM , SonarEncoder
from utils import GloveDataset , add_noise_to_embeddings 
from tqdm.auto import tqdm 
from datasets import load_dataset

# Set random seed for reproducibility
torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--sequence_length', type=int, default=10, help="sequence length for training")
    parser.add_argument('--input_dim', type=int, default=256, help="Input dimension for the model")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension for the model") 
    parser.add_argument('--num_heads', type=int, default=8, help="Number of heads for the model")
    parser.add_argument('--num_layers', type=int, default=6, help="Number of layers for the model")
    parser.add_argument('--ff_dim', type=int, default=2048, help="Feedforward dimension for the model")
    parser.add_argument('--output_dim', type=int, default=256, help="Output dimension for the model")
    parser.add_argument('--epoch', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--noise_level', type=float, default=0.05, help="Noise level for the target")
    parser.add_argument('--vocab_size', type=int, default=5000, help="Vocabulary size for the dataset")
    parser.add_argument('--wandb', type=bool, default=False, help="Use Weights and Biases for logging")
    parser.add_argument('--hf_data', type=str,default=None, help="Path to the Hugging Face dataset")
    parser.add_argument('--dataset_args', type=dict, help="Arguments for the Hugging Face dataset")
    parser.add_argument('--text_column', type=str, default="text", help="Text column in the dataset")
    parser.add_argument('--lang', type=str, default="en", help="Language for the dataset")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (L2 regularization factor)")
    parser.add_argument('--data_sample', type=int, default=1000, help="How much sample to choose from the entire datasets")
    return parser.parse_args()
  

# Centralized device management
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="base-lcm", config=args)
    
    model = BaseLCM(
        input_dim=args.input_dim, 
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers, 
        ff_dim=args.ff_dim, 
        output_dim=args.output_dim
    ).to(device)

    encoder = SonarEncoder(device=device)

 
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # Function to split text into sentences
    def split_into_sentences(text):
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

    # Integrate sentence splitting into the encoding process
    df = load_dataset(args.hf_data, split='train').select(range(args.data_sample))  # For testing

    # Process the text column by splitting into sentences
    print("splitting the corpus into sentences")
    
    processed_texts = []
    for text in tqdm(df[args.text_column], desc="Processing Texts", unit="text"):
      sentences = split_into_sentences(text)
      processed_texts.extend(sentences)

    print("number of sentences,",len(processed_texts))
    # Encode the processed sentences
    input_embeddings = encoder.encode(
        processed_texts, lang=args.lang, batch_size=args.batch_size
    ).to(device)

    
    train_dataset = GloveDataset(input_embeddings, args.sequence_length, args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Target embeddings with noise
    target_embeddings = add_noise_to_embeddings(input_embeddings, args.noise_level).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Training Loop
    for epoch in range(args.epoch):
      model.train()
      running_loss = 0.0

      # Wrapping the dataloader with tqdm for batch progress
      with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epoch}") as pbar:
          for batch_idx, inputs in pbar:
              inputs = to_device(inputs, device)
              batch_targets = target_embeddings[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

              optimizer.zero_grad()
              output_embeddings = model(inputs)
              loss = criterion(output_embeddings, batch_targets)
              loss.backward()
              optimizer.step()

              running_loss += loss.item()

              # Update the tqdm progress bar with the running loss
              pbar.set_postfix(loss=running_loss / (batch_idx + 1))

      # Epoch logging
      epoch_loss = running_loss / len(train_dataloader)
      print(f"Epoch {epoch+1}/{args.epoch} - Loss: {epoch_loss:.4f}")
          
    if args.wandb:
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss})
  
    print("Training Complete!")
    import os 
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "base_lcm_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    if args.wandb:
        wandb.finish()

  
if __name__ == "__main__":
  print("Training the model")
  args = parse_args()
  train(args) 


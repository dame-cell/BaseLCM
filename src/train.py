import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb 
import argparse  
from baselcm import BaseLCM , SonarEncoder
from utils import GloveDataset , compute_accuracy, add_noise_to_embeddings , load_glove_embeddings, prepare_embeddings
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
    parser.add_argument('--file_path', type=str, help="Path to the GloVe embeddings file")
    parser.add_argument('--hf_data', type=str,default=None, help="Path to the Hugging Face dataset")
    parser.add_argument('--dataset_args', type=dict, help="Arguments for the Hugging Face dataset")
    parser.add_argument('--text_column', type=str, default="text", help="Text column in the dataset")
    parser.add_argument('--lang', type=str, default="en", help="Language for the dataset")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (L2 regularization factor)")
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

    if args.hf_data:
        print("Using the Hugging Face dataset")
        df = load_dataset(args.hf_data, split='train').select(range(100))  # For testing
        input_embeddings = encoder.encode(
            df[args.text_column], lang=args.lang, batch_size=args.batch_size
        ).to(device)
    else:
        print("Using GloVe embeddings")
        glove_embeddings = load_glove_embeddings(args.file_path, args.vocab_size)
        input_embeddings = prepare_embeddings(
            glove_embeddings, args.batch_size, args.sequence_length
        ).to(device)
    
    train_dataset = GloveDataset(input_embeddings, args.sequence_length, args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Target embeddings with noise
    target_embeddings = add_noise_to_embeddings(input_embeddings, args.noise_level).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # Training Loop
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        for batch_idx, inputs in enumerate(train_dataloader):
            inputs = to_device(inputs, device)
            batch_targets = target_embeddings[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  
                output_embeddings = model(inputs)
                loss = criterion(output_embeddings, batch_targets)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        
        # Epoch logging
        epoch_loss = running_loss / len(train_dataloader)
        accuracy = compute_accuracy(output_embeddings, batch_targets)
        
        print(f"Epoch {epoch+1}/{args.epoch} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        if args.wandb:
            wandb.log({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": accuracy})
    
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


import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb 
import argparse  
from baselcm import BaseLCM , SonarEncoder
from utils import GloveDataset , compute_accuracy, add_noise_to_embeddings , load_glove_embeddings, prepare_embeddings
from tqdm.auto import tqdm 
import datasets 

# Set random seed for reproducibility
torch.manual_seed(42)
# for training using hugginface dataset you can use the following arguments
# python train.py --hf_data="wikipedia" --dataset_args='{"name":"wikipedia","split":"train"}'

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
    parser.add_argument('--hf_data', type=str, help="Path to the Hugging Face dataset")
    parser.add_argument('--dataset_args', type=dict, help="Arguments for the Hugging Face dataset")
    parser.add_argument('--text_column', type=str, default="text", help="Text column in the dataset")
    parser.add_argument('--lang', type=str, default="en", help="Language for the dataset")
    return parser.parse_args()
  

def train(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if args.wandb:
    wandb.init(project="base-lcm")
    wandb.config.update(args)
  model = BaseLCM(args.input_dim, args.hidden_dim, args.num_heads, args.num_layers, args.ff_dim, args.output_dim,device=device)
  encoder = SonarEncoder()
  
  if args.hf_data:
    print("We will use the Hugging Face dataset")
    df = datasets.load_dataset(args.hf_data,split='train')
    df = df.select(range(1000))
    
    input_embeddings = encoder.encode(df[args.text_column],lang=args.lang,batch_size=args.batch_size)
    input_embeddings = input_embeddings.to(device)
    train_dataset = GloveDataset(input_embeddings, args.sequence_length, args.batch_size)
    
  else:
    print("We will just use the GloVe embeddings")
    #load the glove embeddings
    glove_embeddings = load_glove_embeddings(args.file_path, args.vocab_size)
    input_embeddings = prepare_embeddings(glove_embeddings, args.batch_size, args.sequence_length)
    input_embeddings = input_embeddings.to(device)
    train_dataset = GloveDataset(input_embeddings, args.sequence_length, args.batch_size)
    
  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  print("Number of training batches",len(train_dataloader))
  target_embeddings = add_noise_to_embeddings(input_embeddings, noise_level=args.noise_level)
  target_embeddings = target_embeddings.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  criterion = nn.MSELoss()
  for epoch in range(args.epoch):
    model.train()
    optimizer.zero_grad()
    output_embeddings = model(input_embeddings)
    loss = criterion(output_embeddings, target_embeddings)
    loss.backward()
    optimizer.step()
    
    accuracy = compute_accuracy(output_embeddings, target_embeddings)
    print(f"Epoch {epoch+1}/{args.epoch} Loss: {loss.item()} Accuracy: {accuracy}")
    

  
if __name__ == "__main__":
  print("Training the model")
  args = parse_args()
  train(args) 


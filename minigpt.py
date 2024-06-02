import torch
import torch.nn as nn
from torch.nn import functional as F
import random


class InputEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def get_position_encoding(self, seq_len, n=10000):
        P = torch.zeros((seq_len, embedding_size)).to(device)
        for k in range(seq_len):
            for i in torch.arange(int(embedding_size/2)):
                denominator = torch.pow(n, 2*i/embedding_size)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P


    def forward(self, x):
        out = self.embedding(x)
        P = self.get_position_encoding(x.shape[1])
        f_out = out + P
        return f_out
    

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.keys = nn.Linear(embedding_size, head_size, bias=False)
        self.values = nn.Linear(embedding_size, embedding_size, bias=False)
    
    def forward(self, x):
        q = self.query(x)
        k = self.keys(x)
        v = self.values(x)

        dot_product_matrix = q @ k.transpose(-2, -1)
        upper_triangular_mask = torch.triu(torch.ones_like(dot_product_matrix), diagonal=1).bool()
        dot_product_matrix[upper_triangular_mask] = float('-inf')
        attn_mat = F.softmax(dot_product_matrix / head_size ** 0.5, dim=-1)
        scaled_value_matrix = attn_mat @ v

        return scaled_value_matrix


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, embedding_size*4)
        self.linear2 = nn.Linear(embedding_size*4, embedding_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        f_out = self.linear2(out)
        return f_out


class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleList([SelfAttention() for _ in range(n_heads)])
        self.linear = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        B, T, C = x.shape
        dE = torch.zeros(B, T, C).to(device)
        for attn in self.attn:
            dE += attn.forward(x)
        f_out = self.linear(dE)        
        return f_out
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_attn_block = MultiHeadedAttention() 
        self.feed_forward = FeedForward()
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        out = x + self.multi_attn_block.forward(self.ln1(x))
        out = out + self.feed_forward.forward(self.ln2(out))
        return out
    

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedding = InputEmbeddings()
        self.multi_block = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.ln_f = nn.LayerNorm(embedding_size)

    def forward(self, x):
        B, T = x.shape

        out = self.input_embedding.forward(x)
        out = self.multi_block(out)
        out = self.ln_f(out)
        out = self.linear(out.view(B * T, embedding_size))
        return out
    
    def decode(self, max_tokens=0, tag=None):
        x = [0, 0]
        msg = [0, 0]

        if tag =='max':
            while True:
                tensor_x = torch.tensor(x).unsqueeze(0).to(device)
                out = self.input_embedding.forward(tensor_x)
                out = self.multi_block(out)
                out = self.ln_f(out)
                out = F.softmax(self.linear(out.view(-1, embedding_size)), dim=-1)

                out_idx = torch.multinomial(out, num_samples=1, replacement=True).squeeze(1)[-1]

                x.append(out_idx.item())
                msg.append(out_idx.item())

                if (len(x) >= block_size):
                    x.pop(0)

                print(detokenize(msg))

        for i in range(max_tokens):
            print(f"progress : {(i+1)}/{max_tokens}...", end='\r')

            tensor_x = torch.tensor(x).unsqueeze(0).to(device)
            out = self.input_embedding.forward(tensor_x)
            out = self.multi_block(out)
            out = self.ln_f(out)
            out = F.softmax(self.linear(out.view(-1, embedding_size)), dim=-1)

            out_idx = torch.multinomial(out, num_samples=1, replacement=True).squeeze(1)[-1]
            x.append(out_idx.item())
            msg.append(out_idx.item())
            if len(x) >= block_size:
                x.pop(0)
        return msg
        
        

with open("data.txt", 'r', encoding='utf-8') as f:
    text = f.read()     

chars = sorted(list(set(text)))

tokenize = lambda x : torch.tensor([chars.index(i) for i in list(x)])
detokenize = lambda x : ''.join([chars[i] for i in x])


#--------------------Hyperparameters-----------------------
vocab_size = len(chars)
data_size = len(text)
embedding_size = 64
head_size = 16
n_heads = 4
n_layers = 4
batch_size = 16
steps = 5000
step_chckpt = 100
device = 'cuda'
lr = 1e-3
block_size = 32
#---------------------------------------------------------


transformer = Transformer().to(device)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)
print(sum(p.numel() for p in transformer.parameters())/1e6, 'M parameters')
total_loss = 0

try :
    transformer.load_state_dict(torch.load("model.pth"))
except Exception as e :
    print("no model found")


#(transformer.decode(tag='max')) # TEXT GENERATION
import time
total_time = 0

for k in range(steps):
    start = time.time()
    
    X = []
    y = []

    for i in range(batch_size):
        rdm_idx = random.randint(1, data_size - block_size)
        train_data = text[rdm_idx:rdm_idx+block_size + 1]
        X.append(tokenize(train_data[:block_size]))
        y.append(tokenize(train_data[1:block_size+1]))
    
    X_encode = torch.stack(X).to(device)
    y_encode = torch.stack(y).flatten().to(device)

    output = transformer.forward(X_encode)

    loss = F.cross_entropy(output, y_encode)
    optimizer.zero_grad()
    loss.backward()
    total_loss += loss
    optimizer.step()

    end = time.time()

    total_time += (end - start)
    
    if (k+1) % step_chckpt == 0:
        print(f"steps : {k+1} | loss : {total_loss/step_chckpt} time : {total_time} seconds")
        total_time = 0
        total_loss = 0
        torch.save(transformer.state_dict(), "model.pth")

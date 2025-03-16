import torch

from src.train.model import GPTLanguageModel
from src.train.tokenizier import Rizzler
from torch.nn.utils.rnn import pad_sequence

BATCH_SIZE = 32
BLOCK_SIZE = 32
SPLIT_RATIO = 0.8
LEARNING_RATE = 1e-4
N_EPOCHS = 3000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def split_into_sequences(data, bos_token_id, eos_token_id):
    sequences = []
    current_seq = []

    for token in data:
        if token == bos_token_id and current_seq:
            sequences.append(torch.tensor(current_seq, dtype=torch.long))
            current_seq = [token]
        else:
            current_seq.append(token)
        if token == eos_token_id:
            sequences.append(torch.tensor(current_seq, dtype=torch.long))
            current_seq = []

    if current_seq:
        sequences.append(torch.tensor(current_seq, dtype=torch.long))
    return sequences

def get_batch(sequences, block_size, batch_size, device=DEVICE):
    ix = torch.randint(len(sequences), (batch_size,))
    sampled_sequences = [sequences[i][:block_size] for i in ix]

    # Pad sequences and create mask
    x = pad_sequence(sampled_sequences, batch_first=True, padding_value=0).to(device)
    mask = (x != 0).long()  # 1 for real tokens, 0 for padding

    # Shifted targets (y) are the same sequence shifted by 1
    y = torch.zeros_like(x)
    for i, seq in enumerate(sampled_sequences):
        seq_len = len(seq)
        if seq_len > 1:
            y[i, :seq_len-1] = seq[1:]
    return x, y, mask

def train():
    # Load data and tokenizer
    with open("../data/output.txt", 'r', encoding='utf-8') as file:
        data = file.read()

    Tokenizer = Rizzler()
    Tokenizer.load("../data/tokenizer.json")
    bos_token_id = Tokenizer.tokenizer.token_to_id('[BOS]')
    eos_token_id = Tokenizer.tokenizer.token_to_id('[EOS]')

    # Prepare data
    data = Tokenizer.encode(data)
    sequences = split_into_sequences(data, bos_token_id, eos_token_id)

    n = int(len(sequences) * SPLIT_RATIO)
    train_sequences = sequences[:n]
    val_sequences = sequences[n:]

    # Initialize model
    model = GPTLanguageModel(vocab_size=len(Tokenizer.tokenizer.get_vocab()),
                             n_embed=512, n_head=8, n_layers=12, block_size=BLOCK_SIZE).to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, N_EPOCHS + 1):
        xb, yb, mask = get_batch(train_sequences, BLOCK_SIZE, BATCH_SIZE)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(10):  # Smaller validation steps
                    X, Y, mask = get_batch(val_sequences, BLOCK_SIZE, BATCH_SIZE)
                    _, v_loss = model(X, Y)
                    val_loss += v_loss.item()
            val_loss /= 10

            # Generate sample text
            prompt = "skibidi"
            input_ids = torch.tensor(Tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)
            generated_ids = model.generate(input_ids, max_new_tokens=50, eos_token_id=eos_token_id, top_k=50)
            generated_text = Tokenizer.decode(generated_ids.squeeze(0).tolist())
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")
            print(f"Generated: {generated_text}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "../deploy/model.pth")

            scheduler.step(val_loss)
            model.train()

if __name__ == "__main__":
    train()
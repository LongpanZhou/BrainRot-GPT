from src.train.model import GPTLanguageModel
from src.train.tokenizier import Rizzler
import torch

# Initialize tokenizer and model
tokenizer = Rizzler()
tokenizer.load("../data/tokenizer.json")
vocab_size = len(tokenizer.tokenizer.get_vocab())
model = GPTLanguageModel(vocab_size)

# Example prompt
prompt = "ohio sigma"
input_ids = torch.tensor(tokenizer.encode("[BOS] "+prompt), dtype=torch.long).unsqueeze(0)  # Shape: (1, T)
eos_token_id = tokenizer.tokenizer.token_to_id('[EOS]')

# Generate text
generated_ids = model.generate(input_ids, max_new_tokens=50, eos_token_id=eos_token_id)
print(len(generated_ids.squeeze(0).tolist()))
generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist())
print(generated_text)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit
from tokenizers.processors import TemplateProcessing

# Initialize the tokenizer with BPE model
tokenizer = Tokenizer(BPE())

# Use a pre-tokenizer that splits on whitespace and ensures emojis are separate
# We can combine Whitespace with a custom approach for emojis
tokenizer.pre_tokenizer = Whitespace()

# Define special tokens
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]

# Initialize the trainer with special tokens
trainer = BpeTrainer(special_tokens=special_tokens)

# File path for training data
file_path = ["../data/output.txt"]

# Train the tokenizer
tokenizer.train(file_path, trainer)

# Optional: Add post-processing to handle special tokens in sequences
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [SEP] $B [EOS]",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# Save the tokenizer
tokenizer.save("../data/tokenizer.json")

# Define the Rizzler class
class Rizzler:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
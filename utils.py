import os
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Generates vocabularies for source and target languages from training and validation files
def prepare_vocabularies(train_file, val_file):
    def extract_chars(path, is_source):
        # Extracts characters from the first (source) or second (target) column of the file
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        column = [line.strip().split(maxsplit=1)[0 if is_source else 1] for line in lines]
        return set("".join(column))

    # Collect unique characters from training and validation sets
    src_train_chars = extract_chars(train_file, True)
    tgt_train_chars = extract_chars(train_file, False)
    src_val_chars = extract_chars(val_file, True)
    tgt_val_chars = extract_chars(val_file, False)

    # Combine character sets and prepend special tokens
    combined_src = list(src_train_chars | src_val_chars)
    combined_tgt = list(tgt_train_chars | tgt_val_chars)
    special_tokens = ["<s>", "</s>", "<unk>"]
    combined_src = special_tokens + combined_src
    combined_tgt = special_tokens + combined_tgt

    # Save vocabularies to disk
    os.makedirs("dump", exist_ok=True)
    with open("dump/source_vocab.txt", "w", encoding="utf-8") as src_out, \
         open("dump/target_vocab.txt", "w", encoding="utf-8") as tgt_out:
        src_out.write("\n".join(combined_src))
        tgt_out.write("\n".join(combined_tgt))


# Utility class to convert between characters and indices for a vocabulary
class AlphabetIndexer:
    def __init__(self, column, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            tokens = vocab_file.read().splitlines()

        # Start index from 1 to reserve 0 for padding
        self.str_to_index = {char: idx + 1 for idx, char in enumerate(tokens)}
        self.index_to_str = {v: k for k, v in self.str_to_index.items()}

        # Special token indices
        self.start_tok = self.str_to_index.get("<s>")
        self.end_tok = self.str_to_index.get("</s>")
        self.unk_tok = self.str_to_index.get("<unk>")

        self.entries = column  # pandas Series of text strings
        self.size = len(self.str_to_index) + 1  # +1 because index starts from 1

    def tokenize(self, text):
        # Convert string to list of indices; unknown characters get <unk> index
        return [self.str_to_index.get(ch, self.unk_tok) for ch in text]

    def detokenize(self, tensor):
        # Convert list/tensor of indices back to string
        return "".join([self.index_to_str.get(idx.item(), "<unk>") for idx in tensor])

    def __getitem__(self, index):
        # Get tokenized text with <EOS> token appended
        tokens = self.tokenize(self.entries.iloc[index])
        return tokens + [self.end_tok]


# Custom PyTorch Dataset for transliteration
class TransliterationDataset:
    def __init__(self, file_path, dedup_attestations=False):
        assert file_path.endswith(".tsv"), "Input must be a .tsv file"
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        if dedup_attestations:
            # Deduplicate entries by choosing the one with the highest count
            best_attest = {}
            for row in lines:
                parts = row.strip().split("\t")
                if len(parts) >= 3:
                    src, tgt, count = parts
                    if src not in best_attest or count > best_attest[src]["count"]:
                        best_attest[src] = {"tgt": tgt, "count": count}
            records = [{"source": src, "target": val["tgt"]} for src, val in best_attest.items()]
        else:
            # Default: just extract source and target columns
            records = []
            for row in lines:
                parts = row.strip().split("\t")
                if len(parts) >= 2:
                    records.append({"source": parts[0], "target": parts[1]})

        self.df = pd.DataFrame(records)

        # Initialize vocab indexers using the column data and saved vocab files
        self.source = AlphabetIndexer(self.df["source"], "dump/source_vocab.txt")
        self.target = AlphabetIndexer(self.df["target"], "dump/target_vocab.txt")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Returns (source_tensor, target_tensor) pair for the idx-th item
        src = torch.tensor(self.source[idx], dtype=torch.long)
        tgt = torch.tensor(self.target[idx], dtype=torch.long)
        return src, tgt


# Collate function to pad sequences for batching
def batch_collator(batch):
    # Separates and pads source and target sequences
    src_seqs = [item[0] for item in batch]
    tgt_seqs = [item[1] for item in batch]
    padded_src = pad_sequence(src_seqs, batch_first=True, padding_value=0)
    padded_tgt = pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return padded_src, padded_tgt

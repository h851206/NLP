from attr import fields
import torch
import torch.nn as nn
import torch.optim as optim
import spacy #spacy is the tokenizer
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Create Field for torch text: how the data should be processed
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_en, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k(language_pair=("de", "en"))

# Build the vocabulary used in the dataset: ("Apple", "in", "tree") => (0, 1, 2)
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        embedding_size,
        max_len,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        src_pad_idx
    ) -> None:
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.src_pos_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_pos_embedding = nn.Embedding(max_len, embedding_size)

        self.Transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=forward_expansion,
            dropout=dropout
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_prd_idx = src_pad_idx

    def make_src_mask(self, src):
        # mask out the word which are padding
        # src shape = (src_len, N)
        src_mask =  src.transpose(0,1) == self.src_prd_idx
        # the output is (N, src_len)
        return src_mask

    def forward(
        self,
        src,
        trg
    ):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        # create position tensor which has the same size as src and trg
        # (src_seq_length) -> (src_seq_length, 1) -> (src_seq_length, N)
        src_position = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N) 
        trg_position = torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N)

        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_pos_embedding(src_position)
        )
        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_pos_embedding(trg_position)
        )
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.Transformer.generate_square_subsequent_mask(trg_seq_length)
        
        out = self.Transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask = trg_mask
        )

        out = self.fc_out(self.dropout(out))

        return out
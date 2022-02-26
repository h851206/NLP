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

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

print(type(train_data))
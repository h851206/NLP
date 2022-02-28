import torch
import torch.nn as nn
import torch.optim as optim
import spacy #spacy is the tokenizer
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.datasets import Multi30k 
from torchtext.legacy.data import Field, BucketIterator
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from tqdm import tqdm

spacy_ger = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Create Field for torch text: how the data should be processed
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_en, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

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
        src_pad_idx,
        device
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
        self.device = device

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
        trg_mask = self.Transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        
        out = self.Transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask = trg_mask
        )

        out = self.fc_out(out)

        return out
# Setup the training phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True
save_model = True

# training hyperparameters
num_epochs = 5
learning_rate = 5e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
max_len = 100
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
forward_expansion = 8
dropout = 0.1
src_pad_idx = german.vocab.stoi["<pad>"]

# Tensorboard for nic eplots
writer = SummaryWriter("run/loss_plot")
step = 0

# create iterator(batches)
train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# create the model object
model = Transformer(
    embedding_size=embedding_size,
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    max_len=max_len,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    forward_expansion=forward_expansion,
    dropout=dropout,
    src_pad_idx=src_pad_idx,
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# load model
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "Es ist ein schönes Pferd."

# training seciton
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}")
    
    # save model
    if save_model:
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }

        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device
    )
    print(f"Translated example sentence \n {translated_sentence}")

    with tqdm(train_iterator, unit="batch") as tepoch:

        model.train()
        for batch_idx, batch in enumerate(tepoch):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # forward propagation
            output = model(inp_data, target[:-1])
            # output shape = (num_batch, num words in sentence, voc_size) -> (X, voc_size)
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            writer.add_scalar("Training loss", loss, global_step=step)
            step+=1

            tepoch.set_postfix(loss=loss.item())

# STEPS:

# 1. Specify how preprocessing should be done -> Fields
# 2. Use Dataset to load the data -> TabularDataset (JSON/CSV/TSV Files)
# 3. Construct an iterator to do batching and padding -> BucketIterator

from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import spacy
import torch

# specify tokenizer
spacy_en = spacy.load("en_core_web_sm")
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
# in this data set there are score and quote, but score is not text
quote = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, lower=True)
score = Field(sequential=False, use_vocab=False)

# specify what preprocessing function should be used in each column
fields = {"quote":("q", quote), "score":("s", score)}
# "quote" means the column name in the data. 
# "q" means we can use batch.q to access the data after batches are created. 
# Variable quote is specifying that the preprocessing function for that column

# create the dataset for our own data
train_data, test_data = TabularDataset.splits(
    path="TextExample",
    train="train.json",
    test="test.json",
    format= "json",
    fields= fields
)

# # have an insight
# print(train_data[0].__dict__.keys())
# print(train_data[0].__dict__.values())

# word embedding: tokenized word to integer
quote.build_vocab(train_data, max_size=10000, min_freq=1)
# quote.build_vocab(train_data, max_size=10000, min_freq=1, vectors='glove.6B.100d') #1GB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create the iterator
train_iterator, test_iterator = BucketIterator.splits(
    datasets=(train_data, test_data),
    batch_size=2,
    device=device
)

# for batch in train_iterator:
#     print(batch.q)
#     print(batch.s)


# =============================================================================================
# =============================================================================================

from torchtext.legacy.datasets import Multi30k

spacy_ger = spacy.load("de_core_news_sm")
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
english = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenize_en)
german = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenize_ger)

train_data1, valid_data1, test_data1 = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

german.build_vocab(train_data1, max_size=10000, min_freq=2)
english.build_vocab(train_data1, max_size=10000, min_freq=2)

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data1, valid_data1, test_data1),
    batch_size=32,
    device=device
)
# for batch in train_iterator:
#     print(batch)
print(english.vocab.stoi["the"])
print(english.vocab.itos[5])
print(len(english.vocab))
# =============================================================================================
# =============================================================================================

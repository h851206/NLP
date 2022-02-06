import torch
import torch.nn as nn
from zmq import device


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = (
            embed_size // heads
        )  # split the embed size(word/subword vector) into different heads

        assert (
            self.head_dim * self.heads == self.embed_size
        ), "Embed size need to be div by heads"

        # three individual linear layers before Scaled Dot-Product Attention
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, queries, mask):
        # the shape of input (values, keys, queries) is (batch_size, number of words, embed_size)
        N = queries.shape[0]
        values_len, keys_len, queries_len = (
            values.shape[1],
            keys.shape[1],
            queries.shape[1],
        )

        # split the input into multi head
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        # feed into Linear layer
        values = self.value(values)
        keys = self.key(keys)
        queries = self.query(queries)

        # Multiply queries and keys
        energy = torch.einsum("NQHD, NKHD -> NHQK", [queries, keys])
        # The priority of output dimention is number of batch, head, number of words, and then the number of words it should compare to.
        # queries shape: (N, queries_len, self.heads, self.head_dim)
        # keys shape: (N, keys_len, self.heads, self.head_dim)
        # energy shape: (N, self.heads, queries_len, keys_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float(1e-12))

        # softmax on last dimension
        attention = torch.softmax(energy / (self.head_dim ** (0.5)), dim=3)

        out = torch.einsum("NVHD, NHQK->NQHD", [values, attention]).reshape(
            N, queries_len, self.heads * self.head_dim
        )
        # values shape: (N, values_len, self.heads, self.head_dim)
        # attention shape: (N, self.heads, queries_len, keys_len)
        # out shape (should be the same as queries): (N, queries_len, self.heads, self.head_dim)
        # reshape out = concatenate multi heads

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        x = self.dropout(self.norm1(attention + queries))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + forward))
        # out shape: (batch_size, query length, embed_size)
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        src_voc_size,
        embed_size,
        heads,
        dropout,
        forward_expansion,
        num_layers,
        max_length,
        device,
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.device = device
        self.num_layers = num_layers

        self.word_embedding = nn.Embedding(src_voc_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        Input = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            out = layer(Input, Input, Input, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))

        # Note here we are using src_mask because this is a mask for Scaled Dot-Product Attention
        out = self.transformer(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion,
        num_layer,
        trg_voc_size,
        max_length,
        device,
    ) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layer)
            ]
        )
        self.word_embedding = nn.Embedding(trg_voc_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.fc_out = nn.Linear(embed_size, trg_voc_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        Input = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            out = layer(Input, enc_out, enc_out, src_mask, trg_mask)
            # out shape: (N, seq_length, embed_size)

        out = torch.softmax(self.fc_out(out), dim=2)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_voc_size,
        trg_voc_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        heads=8,
        dropout=0,
        forward_expansion=4,
        num_layer=6,
        max_length=100,
        device="cuda",
    ) -> None:
        super(Transformer, self).__init__()
        self.max_length = max_length
        self.device = device

        self.encoder = EncoderBlock(
            src_voc_size,
            embed_size,
            heads,
            dropout,
            forward_expansion,
            num_layer,
            max_length,
            device,
        )

        self.decoder = Decoder(
            embed_size,
            heads,
            dropout,
            forward_expansion,
            num_layer,
            trg_voc_size,
            max_length,
            device,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # Mask For Scaled Dot-Product Attention: mask out the zero padding part to let them excluded from the attention mechanism
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # Make Mask for masked multi-Head Attention
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encode_out = self.encoder(src, src_mask)
        decode_out = self.decoder(trg, encode_out, src_mask, trg_mask)
        return decode_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device
    ).to(device)
    out = model(x, trg[:, :-1])
    print(out)
    print(out.shape)


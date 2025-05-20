import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import prepare_vocabularies, TransliterationDataset, batch_collator
from dataclasses import dataclass
from collections import namedtuple

# Configuration class to define all model hyperparameters
@dataclass
class ModelParams:
    src_vocab: int = 500
    tgt_vocab: int = 500
    emb_dim: int = 256
    hidden_dim: int = 256
    enc_layers: int = 3
    dec_layers: int = 2
    enc_type: str = "GRU"
    dec_type: str = "GRU"
    bidir_enc: bool = True
    bidir_dec: bool = False
    drop_ratio: float = 0.3
    max_seq_len: int = 32
    start_tok: int = 0
    tf_ratio: float = 0.8
    use_attn: bool = True

    def __post_init__(self):
        assert self.enc_type in ["RNN", "GRU", "LSTM"]
        assert self.dec_type in ["RNN", "GRU", "LSTM"]

# Encoder module: embeds input tokens and passes them through an RNN
class SeqEncoder(nn.Module):
    def __init__(self, cfg: ModelParams):
        super().__init__()
        self.embedding = nn.Embedding(cfg.src_vocab, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_ratio)
        self.rnn = getattr(nn, cfg.enc_type)(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.enc_layers,
            batch_first=True,
            dropout=cfg.drop_ratio if cfg.enc_layers > 1 else 0.0,
            bidirectional=cfg.bidir_enc
        )
        self.final_drop = nn.Dropout(cfg.drop_ratio)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        enc_output, enc_hidden = self.rnn(x)
        enc_output = self.final_drop(enc_output)
        if isinstance(enc_hidden, tuple):
            enc_hidden = enc_hidden[0]  # For LSTM: use only hidden state
        return enc_output, enc_hidden

# Simple decoder without attention
class SimpleDecoder(nn.Module):
    def __init__(self, cfg: ModelParams):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.tgt_vocab, cfg.emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.drop_ratio)
        self.rnn = getattr(nn, cfg.dec_type)(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.dec_layers,
            batch_first=True,
            dropout=cfg.drop_ratio if cfg.dec_layers > 1 else 0.0,
            bidirectional=False
        )
        self.projection = nn.Linear(cfg.hidden_dim, cfg.tgt_vocab)

    def forward(self, enc_output, enc_hidden, target=None, tf_ratio=1.0, beam=1, return_attn=False):
        hidden = enc_hidden
        maxlen = target.shape[1] if target is not None else self.cfg.max_seq_len
        teacher_forcing = torch.rand(1) < tf_ratio
        if not teacher_forcing:
            target = None  # Use model predictions if no teacher forcing

        input_tok = torch.full((enc_output.shape[0], 1), fill_value=self.cfg.start_tok,
                               dtype=torch.long, device=enc_output.device)
        preds = []

        for step in range(maxlen):
            step_out, hidden = self.decode_step(input_tok, hidden)
            preds.append(step_out)
            if target is not None:
                input_tok = target[:, step].unsqueeze(1)
            else:
                input_tok = torch.argmax(F.log_softmax(step_out, dim=-1), dim=-1).detach()

        preds = torch.cat(preds, dim=1)
        return preds, None

    def decode_step(self, input_tok, hidden):
        emb = self.relu(self.embedding(input_tok))
        emb = self.dropout(emb)
        if self.cfg.dec_type == "LSTM":
            hidden = (hidden, None)
        out, hidden = self.rnn(emb, hidden)
        out = self.projection(out)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return out, hidden

# Additive attention mechanism (Bahdanau-style)
class AdditiveAttention(nn.Module):
    def __init__(self, cfg: ModelParams):
        super().__init__()
        enc_mult = 2 if cfg.bidir_enc else 1
        self.linear_e = nn.Linear(cfg.hidden_dim * enc_mult, cfg.hidden_dim)
        self.linear_d = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v = nn.Linear(cfg.hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, enc_states, dec_hidden):
        B, T, H = enc_states.shape
        dec_last = dec_hidden[-1]  # Use the last layer of decoder hidden state
        energy = self.tanh(self.linear_e(enc_states) + self.linear_d(dec_last).unsqueeze(1))
        scores = self.v(energy).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_states)
        return context, attn_weights

# Decoder with attention
class AttnDecoder(nn.Module):
    def __init__(self, cfg: ModelParams):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.tgt_vocab, cfg.emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.drop_ratio)
        self.attn = AdditiveAttention(cfg)
        enc_dim = cfg.hidden_dim * (2 if cfg.bidir_enc else 1)
        rnn_input_dim = cfg.emb_dim + enc_dim
        self.rnn = getattr(nn, cfg.dec_type)(
            input_size=rnn_input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.dec_layers,
            batch_first=True,
            dropout=cfg.drop_ratio if cfg.dec_layers > 1 else 0.0,
            bidirectional=False
        )
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.tgt_vocab)
        self.Hypo = namedtuple("Hypo", ["tokens", "logp", "state"])

    def forward(self, enc_states, enc_hidden, target=None, tf_ratio=1.0, beam=1, return_attn=False):
        return self._greedy(enc_states, enc_hidden, target, tf_ratio, return_attn) if beam == 1 else self._beam(enc_states, enc_hidden, beam)

    def _greedy(self, enc_states, enc_hidden, target, tf_ratio, return_attn):
        hidden = enc_hidden
        B = enc_states.size(0)
        T = target.size(1) if target is not None else self.cfg.max_seq_len
        input_tok = torch.full((B, 1), fill_value=self.cfg.start_tok, dtype=torch.long, device=enc_states.device)
        pred_seq = []
        attn_seq = []

        if torch.rand(1) > tf_ratio:
            target = None  # No teacher forcing

        for t in range(T):
            logits, hidden, attn = self.step(input_tok, hidden, enc_states)
            pred_seq.append(logits)
            if return_attn:
                attn_seq.append(attn)
            if target is not None:
                input_tok = target[:, t].unsqueeze(1)
            else:
                input_tok = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)

        preds = torch.cat(pred_seq, dim=1)
        return preds, torch.stack(attn_seq) if return_attn else None

    def step(self, tok, hidden, enc_states):
        emb = self.relu(self.embedding(tok))
        emb = self.dropout(emb)
        ctx, alpha = self.attn(enc_states, hidden)
        rnn_input = torch.cat([emb, ctx], dim=2)
        if self.cfg.dec_type == "LSTM" and not isinstance(hidden, tuple):
            hidden = (hidden, torch.zeros_like(hidden))
        out, hidden = self.rnn(rnn_input, hidden)
        logits = self.out_proj(out)
        return logits, hidden, alpha

    def _beam(self, enc_states, enc_hidden, k):
        # Placeholder for beam search if needed later
        pass

# Full Seq2Seq model combining encoder and decoder
class Seq2SeqModel(nn.Module):
    def __init__(self, cfg: ModelParams):
        super().__init__()
        self.cfg = cfg
        self.encoder = SeqEncoder(cfg)
        self.decoder = AttnDecoder(cfg) if cfg.use_attn else SimpleDecoder(cfg)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src, tgt=None, tf_ratio=1.0, beam=1, return_attn=False):
        enc_out, enc_hid = self.encoder(src)
        
        # Adjust encoder hidden size for decoder
        expected_layers = self.cfg.dec_layers * (2 if self.cfg.bidir_dec else 1)
        if expected_layers > enc_hid.size(0):
            padding = enc_hid[:expected_layers - enc_hid.size(0)]
            enc_hid = torch.cat((enc_hid, padding), dim=0)
        else:
            enc_hid = enc_hid[:expected_layers]

        if return_attn:
            assert src.size(0) == 1, "Attention maps supported for batch size 1 only"

        preds, attn = self.decoder(enc_out, enc_hid, tgt, tf_ratio, beam, return_attn)

        loss, acc = None, None
        if tgt is not None:
            loss, acc = self._evaluate(preds, tgt)
        return loss, acc, preds, attn

    def _evaluate(self, logits, labels):
        labels = labels.view(-1)
        logits = logits.view(-1, logits.size(-1))
        loss = self.criterion(logits, labels)
        predictions = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
        accuracy = (predictions == labels).float().mean()
        return loss, accuracy

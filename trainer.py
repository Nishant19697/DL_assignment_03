import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
from model import Seq2SeqModel, ModelParams
from utils import batch_collator, TransliterationDataset, prepare_vocabularies  
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os, random, wandb

# Language directory mapping
LANG_DIR_MAP = {
    "hindi": "hi", "bengali": "bn", "gujarati": "gu", "kannada": "kn",
    "malayalam": "ml", "marathi": "mr", "punjabi": "pa", "sindhi": "sd",
    "sinhala": "si", "tamil": "ta", "telugu": "te", "urdu": "ur"
}


@dataclass
class Config:
    language: str = "hindi"
    data_root: str = ""
    batch_sz: int = 256
    n_workers: int = 16
    lr: float = 0.003
    wd: float = 0.0005
    tf_prob: float = 0.8
    n_epochs: int = 10
    emb_dim: int = 256
    hid_dim: int = 256
    enc_layers: int = 3
    dec_layers: int = 2
    enc_type: str = "GRU"
    dec_type: str = "GRU"
    use_bidir: bool = True
    dropout: float = 0.3
    max_seq_len: int = 32
    sos_idx: int = 0
    use_attention: bool = True
    beam_width: int = 1
    enable_wandb: bool = False

    def __post_init__(self):
        if self.language not in LANG_DIR_MAP:
            raise ValueError(f"Language '{self.language}' not supported.")
        lang_code = LANG_DIR_MAP[self.language]
        self.train_path = os.path.join(self.data_root, f"{lang_code}/lexicons/{lang_code}.translit.sampled.train.tsv")
        self.dev_path = os.path.join(self.data_root, f"{lang_code}/lexicons/{lang_code}.translit.sampled.dev.tsv")
        self.test_path = os.path.join(self.data_root, f"{lang_code}/lexicons/{lang_code}.translit.sampled.test.tsv")
        for path in [self.train_path, self.dev_path, self.test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")


class TransliterationTrainer:
    def __init__(self, cfg: Config):
        prepare_vocabularies(cfg.train_path, cfg.dev_path)
        self.cfg = cfg

        train_data = TransliterationDataset(cfg.train_path)
        val_data = TransliterationDataset(cfg.dev_path)
        test_data = TransliterationDataset(cfg.test_path)

        self.train_loader = DataLoader(train_data, batch_size=cfg.batch_sz, shuffle=True, collate_fn=batch_collator,
                                       num_workers=cfg.n_workers, pin_memory=True, persistent_workers=True)
        self.val_loader = DataLoader(val_data, batch_size=cfg.batch_sz, shuffle=False, collate_fn=batch_collator,
                                     num_workers=cfg.n_workers, pin_memory=True, persistent_workers=True)
        self.test_loader = DataLoader(test_data, batch_size=cfg.batch_sz, shuffle=False, collate_fn=batch_collator,
                                      num_workers=cfg.n_workers, pin_memory=True, persistent_workers=True)

        model_cfg = ModelParams(
            decoder_SOS=train_data.target.SOS,
            source_vocab_size=train_data.source.vocab_size,
            target_vocab_size=train_data.target.vocab_size,
            embedding_size=cfg.emb_dim,
            hidden_size=cfg.hid_dim,
            encoder_num_layers=cfg.enc_layers,
            decoder_num_layers=cfg.dec_layers,
            encoder_name=cfg.enc_type,
            decoder_name=cfg.dec_type,
            encoder_bidirectional=cfg.use_bidir,
            dropout_p=cfg.dropout,
            max_length=cfg.max_seq_len,
            teacher_forcing_p=cfg.tf_prob,
            apply_attention=cfg.use_attention
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Seq2SeqModel(model_cfg).to(self.device)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        print(f"Device: {self.device} | Autocast: {self.dtype}")
        print(f"Model Params: {sum(p.numel() for p in self.model.parameters())}")
        print(self.model)

    def run_training(self):
        for ep in range(self.cfg.n_epochs):
            train_loss, train_acc = self._train_epoch(ep)
            val_loss, val_acc = self._validate_epoch(ep)
            print(f"Epoch {ep+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
            if self.cfg.enable_wandb:
                wandb.log({"train_loss": train_loss, "train_accuracy": train_acc,
                           "val_loss": val_loss, "val_accuracy": val_acc})

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, total_acc = 0, 0
        for src, tgt in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            src, tgt = src.to(self.device), tgt.to(self.device)
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                loss, acc, _, _ = self.model(src, tgt, teacher_forcing_p=self.cfg.tf_prob, beam_size=1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_acc += acc.item()
        return total_loss / len(self.train_loader), total_acc / len(self.train_loader)

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss, total_acc = 0, 0
        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc=f"Validating Epoch {epoch+1}"):
                src, tgt = src.to(self.device), tgt.to(self.device)
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    loss, acc, _, _ = self.model(src, tgt, teacher_forcing_p=0.0, beam_size=1)
                total_loss += loss.item()
                total_acc += acc.item()
        return total_loss / len(self.val_loader), total_acc / len(self.val_loader)

    def evaluate(self):
        self.model.eval()
        all_refs, all_preds = [], []
        with torch.no_grad():
            for src, tgt in tqdm(self.test_loader, desc="Running Test"):
                src = src.to(self.device)
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    _, _, pred_seq, _ = self.model(src, teacher_forcing_p=0.0, beam_size=self.cfg.beam_width)
                pred_seq = torch.argmax(F.log_softmax(pred_seq.cpu(), dim=-1), dim=-1)
                all_refs.extend(list(tgt))
                all_preds.extend(list(pred_seq))
        bleu_score, ref_strs, pred_strs = self._calc_bleu(all_refs, all_preds, self.test_loader.dataset)
        exact_match = sum(1 for r, p in zip(ref_strs, pred_strs) if r == p) / len(ref_strs) * 100
        print(f"BLEU Score: {bleu_score:.4f} | Exact Match Accuracy: {exact_match:.2f}%")
        if self.cfg.enable_wandb:
            wandb.log({"bleu_score": bleu_score, "test_accuracy": exact_match})
        with open("results.txt", "w", encoding="utf-8") as f:
            f.write("TARGETS\tPREDICTIONS\n")
            for r, p in zip(ref_strs, pred_strs):
                f.write(f"{r}\t{p}\n")

    def _calc_bleu(self, refs, hyps, testset):
        score, ref_list, hyp_list = 0, [], []
        for ref, hyp in zip(refs, hyps):
            r_trim = ref[:(ref == testset.target.EOS).nonzero(as_tuple=True)[0].item()] if (ref == testset.target.EOS).any() else ref
            h_trim = hyp[:(hyp == testset.target.EOS).nonzero(as_tuple=True)[0].item()] if (hyp == testset.target.EOS).any() else hyp
            r_txt = testset.target.itos(r_trim)
            h_txt = testset.target.itos(h_trim)
            ref_list.append(r_txt)
            hyp_list.append(h_txt)
            score += sentence_bleu([list(r_txt)], list(h_txt), smoothing_function=SmoothingFunction().method1)
        return score / len(refs), ref_list, hyp_list

    def predict(self, word: str, show_attn=False):
        self.model.eval()
        with torch.no_grad():
            token_ids = self.train_loader.dataset.source.stoi(word) + [self.train_loader.dataset.source.EOS]
            x_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            _, _, pred, attn = self.model(x_tensor, teacher_forcing_p=0, beam_size=1, return_attention_map=show_attn)
            pred = torch.argmax(F.log_softmax(pred, dim=-1), dim=-1)[0]
            eos_cutoff = (pred == self.train_loader.dataset.target.EOS).nonzero(as_tuple=True)[0]
            final_seq = pred[:eos_cutoff[0].item()] if len(eos_cutoff) > 0 else pred
            pred_word = self.train_loader.dataset.target.itos(final_seq)
            print(f"Prediction: {pred_word}")
            if show_attn:
                self._plot_attention(attn.squeeze(1, 2)[:, :-1])

    def _plot_attention(self, attn_tensor, save_path="attention_map.png"):
        plt.figure(figsize=(10, 6))
        plt.imshow(attn_tensor.cpu(), aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.title("Attention Map")
        plt.xlabel("Encoder Time Steps")
        plt.ylabel("Decoder Time Steps")
        plt.tight_layout()
        plt.savefig(save_path)

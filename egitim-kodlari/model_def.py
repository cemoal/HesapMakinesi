import torch
import torch.nn as nn
import torchvision.models as models

# Cihaz seçimi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocab mapping (train.py'deki haliyle)
stoi = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3, "!" :4, '"' :5, "#" :6, "&" :7, "'":8, "(" :9, ")" :10,
        "*" :11, "+" :12, "," :13, "-" :14, "--" :15, "---" :16, "." :17, "/" :18, "0" :19}
itos = {v:k for k,v in stoi.items()}

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, max_len=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Encoder: ResNet18 backbone
        resnet = models.resnet18(weights=None)
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)

        self.enc_proj = nn.Linear(512, hidden_dim)

        # Decoder: Transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # Embedding + Positional Encoding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(max_len, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    def _casual_mask(self, size,device):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    def forward(self, images, tgt_seq):
        feats = self.encoder(images)  # [B, C, H, W]
        B, C, H, W = feats.shape
        feats = feats.view(B, C, H * W).permute(0, 2, 1)        # [B, seq_len, C]
        feats = self.enc_proj(feats)                            # [B, seq_len, hidden_dim]
        feats = feats.permute(1, 0, 2)                          # [seq_len_enc, B, dim]
        # Decoder input (+ doğru boyut)
        T = tgt_seq.shape[1]
        pos = self.pos_emb[:T]                                  # [T, dim]
        pos = pos.unsqueeze(0).expand(B, T, -1)                 # [B, T, dim]
        tgt_emb = self.token_emb(tgt_seq) + pos                 # [B, T, dim]
        tgt_emb = tgt_emb.permute(1, 0, 2)                      # [seq_len_dec, B, dim]
        # Causal mask
        tgt_mask = self._casual_mask(T, tgt_emb.device)         # [T, T]

        out = self.decoder(tgt_emb, feats, tgt_mask=tgt_mask)   # [T, B, dim]
        out = out.permute(1, 0, 2)                              # [B, T, dim]
        logits = self.output_layer(out)                         # [B, T, vocab_size]
        return logits

    def greedy_decode(self, img, max_len=100):
        self.eval()
        with torch.no_grad():
            img = img.to(DEVICE)
            seq = [stoi["<sos>"]]

            for _ in range(max_len):
                inp = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
                logits = self.forward(img, inp)
                next_token = logits[0, -1].argmax().item()
                seq.append(next_token)
                if next_token == stoi["<eos>"]:
                    break

        return seq
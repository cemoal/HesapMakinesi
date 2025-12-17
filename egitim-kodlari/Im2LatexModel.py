import torch
import torch.nn as nn
import torchvision.models as models

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, max_len=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Encoder: ResNet18 backbone (son FC katmanı çıkarılıyor)
        resnet = models.resnet18(weights=None)
        modules = list(resnet.children())[:-2]  # son FC ve pooling yok
        self.encoder = nn.Sequential(*modules)

        # Görsel feature'ı projekte et
        self.enc_proj = nn.Linear(512, hidden_dim)

        # Decoder: Transformer
        decoder_layer = nn.TransformerDecokerLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # Embedding + Positional Encoding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(max_len, hidden_dim))

        # Son çıkış: vocab tahmini
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, tgt_seq):
        # Encoder
        feats = self.encoder(images)  # [B, C, H, W]
        B, C, H, W = feats.shape
        feats = feats.view(B, C, H * W).permute(0, 2, 1)  # [B, seq_len, C]
        feats = self.enc_proj(feats)  # [B, seq_len, hidden_dim]
        feats = feats.permute(1, 0, 2)  # Transformer expects [seq_len, B, dim]

        # Decoder input
        tgt_emb = self.token_emb(tgt_seq) + self.pos_emb[:tgt_seq.shape[1]]
        tgt_emb = tgt_emb.permute(1, 0, 2)  # [seq_len, B, dim]

        # Transformer decoder
        out = self.decoder(tgt_emb, feats)  # [seq_len, B, dim]
        out = out.permute(1, 0, 2)  # [B, seq_len, dim]
        logits = self.output_layer(out)  # [B, seq_len, vocab_size]
        return logits

if __name__ == "__main__":
    vocab_size = 500  # şimdilik küçük bir vocab
    model = Im2LatexModel(vocab_size=vocab_size)

    dummy_img = torch.randn(2, 3, 384, 384)   # 2 görsel
    dummy_seq = torch.randint(0, vocab_size, (2, 50))  # 2 formül, 50 token
    out = model(dummy_img, dummy_seq)
    print("Çıkış shape:", out.shape)  # [2, 50, vocab_size]
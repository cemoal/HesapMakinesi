import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import json

from dataset import Im2LatexCSV
from Im2LatexModel import Im2LatexModel
from collate_fn import collate_fn

# -----------------------------# 1. Ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = r"C:\Users\cemal\OneDrive\Desktop\Proje_Egitim\im2latex_train.csv"
VAL_PATH = r"C:\Users\cemal\OneDrive\Desktop\Proje_Egitim\im2latex_validate.csv"
IMAGE_FOLDER = r"C:\Users\cemal\OneDrive\Desktop\Proje_Egitim\formula_images_processed"

# --- Vocab çıkar ---
df = pd.read_csv(CSV_PATH)
formulas = df['formula'].tolist()  # sütun adını kendi dosyana göre ayarla
vocab = set()
for formula in formulas:
    for tok in formula.split():
        vocab.add(tok)

stoi = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
for i, tok in enumerate(sorted(vocab), start=4):
    stoi[tok] = i
itos = {v:k for k,v in stoi.items()}

# JSON'a kaydet (sonradan inference için)
with open("stoi.json", "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)
with open("itos.json", "w", encoding="utf-8") as f:
    json.dump(itos, f, ensure_ascii=False, indent=2)

print("Vocab size:", len(stoi))  # 544

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_losses = []
val_losses = []

# -----------------------------# 2. Veri Yükleyiciler
train_ds = Im2LatexCSV(CSV_PATH, IMAGE_FOLDER, stoi)
val_ds = Im2LatexCSV(VAL_PATH, IMAGE_FOLDER, stoi)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, pad_idx=stoi["<pad>"]))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_idx=stoi["<pad>"]))

# -----------------------------# 3. Model, Kayıp Fonksiyonu ve Optimizatör
model = Im2LatexModel(vocab_size=len(stoi)).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Basit bir StepLR örneği
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# Model, optimizer, scheduler tanımlandıktan sonra
model.load_state_dict(torch.load("checkpoint_epoch8.pth", map_location=device))

start_epoch = 9  # 8 bitti, 9'dan devam


# -----------------------------# 4. Eğitim Döngüsü
start_epoch = 9
for epoch in range(start_epoch, EPOCHS+1):
    model.train()
    total_loss = 0

    for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch"):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)

        logits = model(imgs, tgts[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        

    torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")
    torch.save(optimizer.state_dict(), f"optimizer_epoch{epoch+1}.pth")
    print(f"Checkpoint saved for epoch {epoch+1}")

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            logits = model(imgs, tgts[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    tqdm.write(f"Epoch {epoch}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    train_losses.append(avg_loss)
    val_losses.append(val_loss)

# -----------------------------# 5. Kayıp Grafiği
plt.figure(figsize=(8,5))
plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, EPOCHS+1), val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
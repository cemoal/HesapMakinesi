import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re

def latex_tokenize(s: str):
    s = s.strip()
    tokens = re.findall(r'(\\[A-Za-z]+)|[{}_^]|[\[\]\(\)]|[0-9]+|[A-Za-z]+|[^ \t\n]', s)
    return tokens
# kendi yolunuza göre yapmanız lazım
CSV_PATH = r"C:\Users\cemal\OneDrive\Desktop\Proje_Egitim\im2latex_validate.csv"
IMAGE_FOLDER = r"C:\Users\cemal\OneDrive\Desktop\Proje_Egitim\formula_images_processed"

class Im2LatexCSV(Dataset):
    def __init__(self, csv_path, image_folder, stoi, max_len=256):
        self.image_folder = image_folder
        self.stoi = stoi
        self.max_len = max_len

        df = pd.read_csv(csv_path)
        self.items = []
        for _, row in df.iterrows():
            latex = row["formula"].strip()
            img_name = row["image"].strip()
            img_path = os.path.join(image_folder, img_name)
            if os.path.exists(img_path):
                self.items.append((img_path, latex))

        self.tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def encode(self, s):
        toks = latex_tokenize(s)
        ids = [self.stoi["<sos>"]] + [self.stoi.get(t, self.stoi["<unk>"]) for t in toks] + [self.stoi["<eos>"]]
        ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, i):
        img_path, latex = self.items[i]
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        tgt = self.encode(latex)
        return img, tgt

    def __len__(self):
        return len(self.items)

if __name__ == "__main__":
    stoi = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
    ds = Im2LatexCSV(CSV_PATH, IMAGE_FOLDER, stoi)
    print("Dataset uzunluğu:", len(ds))
    if len(ds) > 0:
        img, tgt = ds[0]
        print("Görsel shape:", img.shape)
        print("Token dizisi uzunluğu:", tgt.shape)
        print("İlk örnek:", ds.items[0])

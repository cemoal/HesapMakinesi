import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_idx=0):
    """
    batch: list of (image, tgt) örnekleri
    pad_idx: <pad> token ID
    """
    imgs, tgts = zip(*batch)  # batch içinden ayır

    # Görselleri stackle
    imgs = torch.stack(imgs, dim=0)

    # Target dizilerini padle
    tgts = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)

    return imgs, tgts
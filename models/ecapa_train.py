# ecapa_train.py
import os
import random
import math
from glob import glob
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

####################
# Config / Helpers #
####################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS = 80
SEGMENT_SECONDS = 3.0   # crop length during training
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
EMB_DIM = 192  # embedding dimension
NUM_WORKERS = 4
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

####################
# Dataset
####################
def load_audio(path, sr=SAMPLE_RATE):
    wav, r = librosa.load(path, sr=sr, mono=True)
    return wav

def compute_log_mel(wav):
    # wav: 1-d numpy float32
    # produce shape [n_mels, time]
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,    # 10 ms
        win_length=400,    # 25 ms
        n_mels=N_MELS,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # normalize per-utterance
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    return log_mel.astype(np.float32)

class SpeakerDataset(Dataset):
    def __init__(self, root: str, seg_samples=SEGMENT_SAMPLES):
        self.root = Path(root)
        self.samples = []
        speakers = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.le = LabelEncoder()
        self.le.fit(speakers)
        for sp in speakers:
            files = list((self.root/sp).glob("*.wav"))
            for f in files:
                self.samples.append((str(f), sp))
        self.seg_samples = seg_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sp = self.samples[idx]
        wav = load_audio(path)
        # if shorter than segment, pad
        if len(wav) < self.seg_samples:
            pad = self.seg_samples - len(wav)
            wav = np.pad(wav, (0, pad), mode="wrap")   # wrap or constant
        # random crop
        start = random.randint(0, max(0, len(wav)-self.seg_samples))
        crop = wav[start:start+self.seg_samples]
        feats = compute_log_mel(crop)  # shape [n_mels, time]
        # optionally add augmentation here
        label = int(self.le.transform([sp])[0])
        # to tensor -> [1, n_mels, time]
        return torch.from_numpy(feats).unsqueeze(0), label

####################
# ECAPA-TDNN Blocks (compact)
####################
class Conv1dAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels//r, 1)
        self.fc2 = nn.Conv1d(channels//r, channels, 1)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.sig(self.fc2(s))
        return x * s

class Res2Block(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.conv1 = nn.Conv1d(self.width, self.width, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.bn = nn.BatchNorm1d(self.width)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T]
        splits = torch.split(x, self.width, dim=1)
        out_splits = []
        for i, sp in enumerate(splits):
            if i == 0:
                out = sp
            else:
                out = sp + out
                out = self.act(self.bn(self.conv1(out)))
            out_splits.append(out)
        return torch.cat(out_splits, dim=1)

class SE_Res2Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv1dAct(channels, channels, kernel=1)
        self.res2 = Res2Block(channels)
        self.conv2 = Conv1dAct(channels, channels, kernel=1)
        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.res2(out)
        out = self.conv2(out)
        out = self.se(out)
        return out + x

class StatsPooling(nn.Module):
    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, emb_dim=EMB_DIM, num_classes=None):
        super().__init__()
        # input expected shape [B, 1, n_mels, time] -> we collapse freq into channels for 1D conv
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        self.layer2 = SE_Res2Block(channels)
        self.layer3 = SE_Res2Block(channels)
        self.layer4 = SE_Res2Block(channels)
        self.conv_cat = nn.Conv1d(channels*3, channels, kernel_size=1)
        self.bn_cat = nn.BatchNorm1d(channels)
        self.relu_cat = nn.ReLU()
        self.pooling = StatsPooling()
        self.fc = nn.Linear(channels*2, emb_dim)
        self.bn_fc = nn.BatchNorm1d(emb_dim)
        # optional classifier
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, get_embedding=False):
        # x: [B, 1, n_mels, time]
        B, _, F, T = x.shape
        x = x.squeeze(1)                # [B, F, T]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        cat = torch.cat([out2, out3, out4], dim=1)
        cat = self.relu_cat(self.bn_cat(self.conv_cat(cat)))
        stats = self.pooling(cat)       # [B, channels*2]
        emb = self.bn_fc(self.fc(stats))  # embedding
        if get_embedding:
            return emb
        if self.classifier is not None:
            logits = self.classifier(emb)
            return logits
        return emb

####################
# Training loop
####################
def train(train_root):
    # build dataset
    ds = SpeakerDataset(train_root)
    num_classes = len(ds.le.classes_)
    print("Num speakers:", num_classes)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    model = ECAPA_TDNN(in_channels=N_MELS, channels=512, emb_dim=EMB_DIM, num_classes=num_classes)
    model.to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(dl), total=len(dl))
        for i, (feats, labels) in pbar:
            # feats: [B, 1, n_mels, time]
            feats = feats.to(DEVICE)   # float32
            labels = labels.to(DEVICE)
            # forward
            logits = model(feats)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} loss {running_loss/(i+1):.4f}")
        scheduler.step()

        # save checkpoint
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "le_classes": ds.le.classes_.tolist()}
        torch.save(ckpt, f"ecapa_epoch{epoch}.pt")

    # save final encoder for embedding extraction
    torch.save({"model": model.state_dict(), "le": ds.le.classes_.tolist()}, "ecapa_final.pt")
    print("Training finished.")

####################
# Embedding extraction & simple eval
####################
def extract_embeddings(model_path: str, wav_paths: List[str]):
    ck = torch.load(model_path, map_location="cpu")
    classes = ck.get("le") or ck.get("le_classes")
    # create model without classifier
    model = ECAPA_TDNN(in_channels=N_MELS, channels=512, emb_dim=EMB_DIM, num_classes=None)
    model.load_state_dict(ck["model"], strict=False)
    model.to(DEVICE)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for p in wav_paths:
            wav = load_audio(p)
            # pad to min length
            if len(wav) < SEGMENT_SAMPLES:
                wav = np.pad(wav, (0, SEGMENT_SAMPLES - len(wav)), mode="wrap")
            feats = compute_log_mel(wav[:SEGMENT_SAMPLES])
            tensor = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,F,T]
            emb = model(tensor, get_embedding=True)  # [1, emb_dim]
            emb = emb.cpu().numpy().squeeze()
            # L2 normalize
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            embeddings.append((p, emb))
    return embeddings

def simple_identification_demo(embeddings: List[Tuple[str, np.ndarray]], labels: List[str]):
    # For demonstration: compute centroid per label, then nearest-centroid classification
    from collections import defaultdict
    lab_embs = defaultdict(list)
    for (path, emb), lab in zip(embeddings, labels):
        lab_embs[lab].append(emb)
    centroids = {lab: np.mean(np.stack(v), axis=0) for lab, v in lab_embs.items()}
    # Evaluate by leaving one out etc. This is a stub; adapt with real splits.
    return centroids

####################
# CLI
####################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default=None, help="root with speaker subfolders")
    parser.add_argument("--mode", type=str, choices=["train", "extract"], default="train")
    parser.add_argument("--wav", type=str, nargs="*", help="wav files for embedding extraction")
    parser.add_argument("--model", type=str, default="ecapa_final.pt")
    args = parser.parse_args()
    if args.mode == "train":
        assert args.train_root, "provide --train_root"
        train(args.train_root)
    else:
        assert args.wav, "provide wav files to extract"
        embs = extract_embeddings(args.model, args.wav)
        for p, e in embs:
            print(p, e.shape)


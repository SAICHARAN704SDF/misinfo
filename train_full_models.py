import os
import random
import warnings
import zipfile
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --------------------------
# Config
# --------------------------
BASE = os.getcwd()
DATA_DIR = os.path.join(BASE, "data")

MIS_FILE = os.path.join(DATA_DIR, "misinfo_train.csv")
NON_FILE = os.path.join(DATA_DIR, "nonmisinfo_train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_data_without_label.csv")

MODEL_NAME = "microsoft/mdeberta-v3-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
FP16 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(BASE, "model_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device -> {DEVICE}")

# --------------------------
# Utilities
# --------------------------
def read_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)

def save_zip(files, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if os.path.exists(f):
                zf.write(f, arcname=os.path.basename(f))

# --------------------------
# Load & merge
# --------------------------
print("Loading CSVs...")
df_mis = read_csv_safe(MIS_FILE)
df_non = read_csv_safe(NON_FILE)
print("Shapes:", df_mis.shape, df_non.shape)

for df in (df_mis, df_non):
    if 'id' not in df.columns:
        raise KeyError("Each CSV must have an 'id' column.")
    if 'text' not in df.columns:
        raise KeyError("Each CSV must have a 'text' column.")
    if 'label' not in df.columns:
        raise KeyError("Each CSV must have a 'label' column.")

# Keep IDs as string to avoid scientific notation
df_mis['id'] = df_mis['id'].astype(str)
df_non['id'] = df_non['id'].astype(str)

df_train = pd.concat([df_mis, df_non], ignore_index=True).reset_index(drop=True)
df_train['text'] = df_train['text'].astype(str).fillna("").str.strip()
df_train = df_train[df_train['text'] != ""]

label_map = {'misinfo': 1, 'nonmisinfo': 0}
df_train['label_num'] = df_train['label'].map(label_map)
print("Combined train distribution:", df_train['label'].value_counts().to_dict())

# --------------------------
# Train/Validation split
# --------------------------
train_texts, val_texts, train_ids, val_ids, y_train, y_val = train_test_split(
    df_train['text'], df_train['id'], df_train['label_num'],
    test_size=0.15, stratify=df_train['label_num'], random_state=42
)

train_df = pd.DataFrame({'id': train_ids.reset_index(drop=True),
                         'text': train_texts.reset_index(drop=True),
                         'label': y_train.reset_index(drop=True)})

# --------------------------
# Oversample minority
# --------------------------
print("Original training class counts:", train_df['label'].value_counts().to_dict())
counts = train_df['label'].value_counts()
if len(counts) > 1:
    maj_label = counts.idxmax()
    min_label = counts.idxmin()
    n_maj = counts.max()
    df_min = train_df[train_df['label'] == min_label]
    df_maj = train_df[train_df['label'] == maj_label]
    reps = n_maj // len(df_min)
    remainder = n_maj % len(df_min)
    df_min_oversampled = pd.concat(
        [df_min] * reps + [df_min.sample(n=remainder, replace=False, random_state=42)],
        ignore_index=True
    )
    train_df = pd.concat([df_maj, df_min_oversampled], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print("After oversampling:", train_df['label'].value_counts().to_dict())

# --------------------------
# Tokenizer and Dataset
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, ids=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.ids is not None:
            item['id'] = self.ids[idx]
        return item

train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(),
                            train_df['id'].tolist(), tokenizer, MAX_LEN)
val_dataset = TextDataset(val_texts.reset_index(drop=True).tolist(),
                          y_val.reset_index(drop=True).tolist(),
                          val_ids.reset_index(drop=True).tolist(),
                          tokenizer, MAX_LEN)

# --------------------------
# Model + optimizer
# --------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)
scaler = torch.cuda.amp.GradScaler() if (FP16 and DEVICE.type == 'cuda') else None

# --------------------------
# Train + eval functions
# --------------------------
def evaluate(model, loader):
    model.eval()
    preds, labs, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch['id'])
            inputs = {k: batch[k].to(DEVICE) for k in ['input_ids', 'attention_mask', 'token_type_ids'] if k in batch}
            if 'labels' in batch:
                labs.extend(batch['labels'].cpu().numpy().tolist())
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
            preds.extend(probs.tolist())
    return ids, preds, labs

best_val_f1 = -1.0
print("Start training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs = {k: batch[k].to(DEVICE) for k in ['input_ids', 'attention_mask', 'token_type_ids'] if k in batch}
        labels = batch['labels'].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

    val_ids, val_probs, val_labels = evaluate(model, val_loader)
    val_preds = [1 if p > 0.5 else 0 for p in val_probs]
    f1 = f1_score(val_labels, val_preds)
    print(f"Val F1 = {f1:.4f}")
    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        print("✅ Best model saved")

# --------------------------
# Final validation predictions
# --------------------------
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=DEVICE))
model.to(DEVICE)

val_ids, val_probs, val_labels = evaluate(model, val_loader)
thresholds = np.arange(0.1, 0.9, 0.01)
best_t, best_f1 = max([(t, f1_score(val_labels, [1 if p>t else 0 for p in val_probs])) for t in thresholds], key=lambda x: x[1])
val_preds_final = [1 if p>best_t else 0 for p in val_probs]

# Metrics
precision = precision_score(val_labels, val_preds_final, zero_division=0)
recall = recall_score(val_labels, val_preds_final, zero_division=0)
f1 = f1_score(val_labels, val_preds_final, zero_division=0)
print(f"Validation metrics -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Save validation CSV
val_ids_str = [str(i) for i in val_ids]
df_val_out = pd.DataFrame({'id': val_ids_str, 'label': ['misinfo' if v==1 else 'nonmisinfo' for v in val_preds_final]})
val_path = os.path.join(BASE, "validation.csv")
df_val_out.to_csv(val_path, index=False)
print(f"✅ Saved validation.csv -> {val_path}")
print("Sample validation IDs:", val_ids_str[:5])
print("Sample validation labels:", df_val_out['label'].tolist()[:5])

# --------------------------
# Test predictions
# --------------------------
df_test = read_csv_safe(TEST_FILE)
df_test['text'] = df_test['text'].astype(str).fillna("").str.strip()
df_test['id'] = df_test['id'].astype(str)  # keep as string
test_dataset = TextDataset(df_test['text'].tolist(), labels=None, ids=df_test['id'].tolist(), tokenizer=tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

test_ids, test_probs, _ = evaluate(model, test_loader)
test_preds_final = [1 if p>best_t else 0 for p in test_probs]
test_ids_str = [str(i) for i in test_ids]

df_sub = pd.DataFrame({'id': test_ids_str, 'label': ['misinfo' if v==1 else 'nonmisinfo' for v in test_preds_final]})
sub_path = os.path.join(BASE, "submission.csv")
df_sub.to_csv(sub_path, index=False)
save_zip([sub_path], os.path.join(BASE, "submission.zip"))
print(f"✅ Saved submission.csv -> {sub_path}")
print("Sample test IDs:", test_ids_str[:5])
print("Sample test labels:", df_sub['label'].tolist()[:5])

print("All done. Best val F1:", best_val_f1)
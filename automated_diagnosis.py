import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import color, filters, measure, morphology
import kagglehub

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline


## this is for the cnn

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.utils.class_weight import compute_class_weight

 
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
 
def save_fig(name, dpi=300, exts=("png", "pdf")):
    plt.tight_layout()
    for e in exts:
        plt.savefig(os.path.join(FIGDIR, f"{name}.{e}"), dpi=dpi, bbox_inches="tight")
    print(f"saved {name} ({', '.join(exts)})")


path_meta = kagglehub.dataset_download(
    "kmader/skin-cancer-mnist-ham10000", 
          path="HAM10000_metadata.csv")

meta = pd.read_csv(path_meta)

data_root = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
 
img_dirs = [
    os.path.join(data_root, "HAM10000_images_part_1"),
    os.path.join(data_root, "HAM10000_images_part_2"),
]
 
def find_image_path(image_id):
    for d in img_dirs:
        p = os.path.join(d, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None
 
meta["image_path"] = meta.image_id.apply(find_image_path)
print(meta.image_path.isna().sum(), "images not found")
meta = meta.dropna(subset=["image_path"]).reset_index(drop=True)



def extract_features(image_path, size=128):
    img = np.array(Image.open(image_path).convert("RGB").resize((size, size)))
    gray = color.rgb2gray(img)

    mask = gray < filters.threshold_otsu(gray)
    mask = morphology.remove_small_objects(mask, min_size=50)
    mask = morphology.remove_small_holes(mask, area_threshold=50)

    labeled = measure.label(mask)
    props = measure.regionprops(labeled, intensity_image=img)

    if not props:
        return None  # segmentation failed

    r = max(props, key=lambda p: p.area)  # largest region = the lesion

    perimeter = max(r.perimeter, 1.0)
    flipped = np.fliplr(r.image)
    asymmetry = np.logical_xor(r.image, flipped).mean() if flipped.shape == r.image.shape else np.nan

    return {
        "area": r.area,
        "perimeter": perimeter,
        "eccentricity": r.eccentricity,
        "border_irregularity": perimeter**2 / (4 * np.pi * r.area),
        "diameter": 2 * np.sqrt(r.area / np.pi),
        "asymmetry": asymmetry,
        "r_mean": r.intensity_mean[0], "g_mean": r.intensity_mean[1], "b_mean": r.intensity_mean[2],
    }



FEATURES_CACHE = "image_features.csv"

if os.path.exists(FEATURES_CACHE):
    feat_df = pd.read_csv(FEATURES_CACHE)
else:
    records = []
    for row in meta.itertuples():
        if row.Index % 500 == 0:
            print(f"{row.Index}/{len(meta)}")
        f = extract_features(row.image_path)
        records.append(f)
    feat_df = pd.DataFrame(records, index=meta.index)
    feat_df.to_csv(FEATURES_CACHE, index=False)

df = pd.concat([meta, feat_df], axis=1)

df = pd.concat([meta, feat_df], axis=1)
df = df.loc[:, ~df.columns.duplicated()]



gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df.lesion_id))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_temp = df.iloc[temp_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp.lesion_id))
df_val = df_temp.iloc[val_idx].reset_index(drop=True)
df_test = df_temp.iloc[test_idx].reset_index(drop=True)

print(f"train {len(df_train)}, val {len(df_val)}, test {len(df_test)}")



numeric_cols = ["area", "perimeter", "eccentricity", "border_irregularity",
                "diameter", "asymmetry", "r_mean", "g_mean", "b_mean", "age"]
categorical_cols = ["sex", "localization"]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

rf_pipeline = ImbPipeline([
    ("prep", preprocess),
    ("ros", RandomOverSampler(random_state=42)),
    ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     random_state=42, n_jobs=-1, criterion='gini')),
])

X_train = df_train[numeric_cols + categorical_cols]
y_train = df_train["dx"]
rf_pipeline.fit(X_train, y_train)



X_val = df_val[numeric_cols + categorical_cols]
y_val = df_val["dx"]
val_pred = rf_pipeline.predict(X_val)

print("Balanced accuracy:", balanced_accuracy_score(y_val, val_pred))
print("Macro-F1:", f1_score(y_val, val_pred, average="macro"))
print(classification_report(y_val, val_pred))



fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_val, val_pred, ax=ax,
    xticks_rotation=45, colorbar=False, normalize="true",
)
plt.title("Random Forest — validation confusion matrix (row-normalised)")
save_fig("rf_confusion_matrix")
plt.show()



## this is for the cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

classes = sorted(df.dx.unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
print(classes)

IMG_SIZE = 224

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class LesionDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df.image_path.values
        self.labels = df.dx.map(class_to_idx).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

train_ds = LesionDataset(df_train, train_tf)
val_ds = LesionDataset(df_val, eval_tf)
test_ds = LesionDataset(df_test, eval_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=4)


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)




weights = compute_class_weight(
    "balanced", classes=np.arange(len(classes)), y=df_train.dx.map(class_to_idx)
)
weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), all_preds, all_labels

for epoch in range(20):
    train_loss, _, _ = run_epoch(train_loader, training=True)
    val_loss, val_preds, val_labels = run_epoch(val_loader, training=False)
    print(f"epoch {epoch+1}: train_loss={train_loss:.3f} val_loss={val_loss:.3f}")


print("Balanced accuracy:", balanced_accuracy_score(val_labels, val_preds))
print("Macro-F1:", f1_score(val_labels, val_preds, average="macro"))
print(classification_report(val_labels, val_preds, target_names=classes))


fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    val_labels, val_preds, display_labels=classes, ax=ax,
    xticks_rotation=45, colorbar=False, normalize="true",
)
plt.title("CNN — validation confusion matrix (row-normalised)")
save_fig("cnn_confusion_matrix")
plt.show()

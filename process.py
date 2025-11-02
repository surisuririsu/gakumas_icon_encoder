import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

ENTITY_TYPES = ["p_item", "skill_card"]

# Copy icon images to dataset
for entity_type in ENTITY_TYPES:
    os.makedirs(f"data/{entity_type}s", exist_ok=True)
    for filename in os.listdir(f"gk-img/docs/{entity_type}s/icons"):
        id = os.path.splitext(filename)[0]
        cls_dir = os.path.join(f"data/{entity_type}s", id)
        os.makedirs(cls_dir, exist_ok=True)
        shutil.copy(
            os.path.join(f"gk-img/docs/{entity_type}s/icons", filename),
            os.path.join(cls_dir, "icon.webp"),
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10


class SmallEmbeddingNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → 1×1 output
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        emb = self.embedding(x)
        out = self.classifier(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return emb, out


class ClassifierNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → 1×1 output
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        emb = self.embedding(x)
        out = self.classifier(emb)
        return out


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResize(16, 64, antialias=False),
        v2.Resize((64, 64)),
        v2.RandomResizedCrop(64, scale=(0.9, 1.0)),
        v2.ColorJitter(0.2, 0.2, 0.2),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def train(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings, outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)


for entity_type in ENTITY_TYPES:
    data_dir = f"./data/{entity_type}s"
    model_save_path = f"./{entity_type}_model.pt"
    onnx_save_path = f"./{entity_type}_model.onnx"

    print(data_dir)

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(dataset.classes)
    print(f"Number of classes for {entity_type}: {num_classes}")

    model = SmallEmbeddingNet(num_classes, EMBEDDING_DIM).to(device)

    old_state = torch.load(model_save_path, map_location="cpu")
    new_state = model.state_dict()
    for k in new_state.keys():
        if k in old_state and "classifier" not in k:
            new_state[k] = old_state[k]

    old_w = old_state["classifier.weight"]
    old_b = old_state["classifier.bias"]

    new_w = new_state["classifier.weight"]
    new_b = new_state["classifier.bias"]

    new_w[: old_w.shape[0]] = old_w
    new_b[: old_b.shape[0]] = old_b

    new_state["classifier.weight"] = new_w
    new_state["classifier.bias"] = new_b

    model.load_state_dict(new_state)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    while True:
        loss = train(model, dataloader, optimizer, EPOCHS)
        if loss < 0.2:
            print("Training complete with acceptable loss.")
            break

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    model = ClassifierNet(num_classes, EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location="cpu"), strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)
    dummy_input = dummy_input.to(device)
    torch.onnx.export(
        model.eval(),
        dummy_input,
        onnx_save_path,
        input_names=["input"],
        output_names=["classifier"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"ONNX model saved to {onnx_save_path}")

    image_dir = f"gk-img/docs/{entity_type}s/icons"

    classes = []

    with torch.no_grad():
        for filename in sorted(os.listdir(image_dir)):
            card_id = os.path.splitext(filename)[0]
            classes.append(card_id)

    with open(f"{entity_type}_classes.json", "w") as f:
        json.dump(classes, f)

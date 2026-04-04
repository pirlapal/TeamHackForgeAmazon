"""Quick test to verify DL training is working"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from libraries.dl.train import train_epoch, evaluate

# Load tiny dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
texts = ["This is great!", "This is terrible!", "I love it", "I hate it"] * 10
labels = [1, 0, 1, 0] * 10

encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
packed = torch.stack([encoded["input_ids"].long(), encoded["attention_mask"].long()], dim=1)
labels_t = torch.tensor(labels, dtype=torch.long)

train_loader = DataLoader(TensorDataset(packed, labels_t), batch_size=8, shuffle=True)

# Simple model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("Training for 3 epochs...")
for epoch in range(3):
    loss = train_epoch(model, train_loader, criterion, optimizer, device="cpu")
    result = evaluate(model, train_loader, criterion, device="cpu")
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={result['accuracy']:.1%}")

print(f"\nFinal accuracy: {result['accuracy']:.1%}")
if result['accuracy'] > 0.5:
    print("✓ Model is learning!")
else:
    print("✗ Model not learning - still at issue")

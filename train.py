import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
from dataset import PunctuationDataset
from model import PunctuationRestorationModel
import torch.nn as nn

def train():
    # Initialize tokenizer, dataset, and dataloader
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    dataset = PunctuationDataset("train_dataset.xlsx", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = PunctuationRestorationModel(num_labels=len(dataset.label2idx)).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    min_loss = float('inf')  # Initialize a variable to track the minimum loss

    # Training loop
    for epoch in range(30):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/30, Loss: {avg_loss:.4f}')

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), "best-model-checkpoint.pth")
            print(f'Checkpoint saved at Epoch {epoch+1} with Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    train()

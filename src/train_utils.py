import torch
from tqdm import tqdm

def train_model(model, train_loader, device, optimizer, criterion, num_epochs=20, save_path="model.pth"):
    model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0.0

        for img, mask in train_loader:
            img = img.float().to(device)
            mask = mask.float().to(device).unsqueeze(1)  # (B, 1, H, W)

            y_pred = model(img)
            loss = criterion(y_pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

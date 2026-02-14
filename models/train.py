import torch
from torch.utils.data import DataLoader
from dataset import OpticalFlowSequenceDataset
from convlstm import ConvLSTMAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLOW_PATH = "data/flow/cam1"   # start with ONE camera
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-3

dataset = OpticalFlowSequenceDataset(FLOW_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ConvLSTMAutoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    total_loss = 0

    for seq in loader:
        seq = seq.to(DEVICE)

        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "convlstm_autoencoder.pth")

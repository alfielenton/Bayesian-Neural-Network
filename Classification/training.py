import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from data_handler import Handler
from network import Network
from saver import Saver

h = Handler()
saver = Saver()

device = "cuda"
split = 0.85
n_excluded = 19
excluded = random.sample(h.targets, n_excluded)
lr = .05
alpha = 1e-8

print(f"\nCollecting data...\nExcluded birds: {excluded}\n")
paths, targets = h.gather_data(excluded)
N = len(paths)
C = len(targets)
num_train = int(split * N)
num_test = N - num_train

print("Targets --\n")
for p in targets:
    print("\t"+p)
saver.save_targets(targets)

print(f"\nTotal data: {N}")
print(f"Total classes: {C}\n\nNumber training: {num_train}\nNumber testing: {num_test}")

train_data = random.sample(paths, k=num_train)
test_data = [p for p in paths if p not in train_data]

NUM_EPOCHS = 100
batch_size = 64

model = Network(num_channels=3, num_classes=C).to(device)
optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
criterion = nn.CrossEntropyLoss()
train_dl = DataLoader(paths, batch_size=batch_size, shuffle=True)

print(f"\nNumber of epochs: {NUM_EPOCHS}\n")

for epoch in range(NUM_EPOCHS):

    print(f"Epoch {epoch + 1}--\n")
    for n_batch, batch in enumerate(train_dl):

        x_batch, y_batch = h.build_batch(batch)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        y_hat_max = y_hat.max(dim=1).indices
        num_corr = (y_hat_max == y_batch).sum()
        acc = num_corr / y_batch.size(0)
        print(f"\tBatch {n_batch + 1} | Loss {loss.item():.3f} | Accuracy {100 * acc:.3f}%")

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        saver.save_results(epoch, loss.item(), acc.item())
    saver.save_model(model.state_dict())
    print("\nModel saved!\n")
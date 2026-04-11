import matplotlib.pyplot as plt

# Your losses (already from training)
train_losses = [2.39,2.28,1.94,1.87,1.76,1.79,1.68,1.63,1.57]
val_losses = [2.24,2.17,2.14,2.10,1.95,1.93,1.81,1.92,1.80]

epochs = range(1, len(train_losses)+1)

plt.figure()

plt.plot(epochs, train_losses, 'o-', label="Train Loss")
plt.plot(epochs, val_losses, 's-', label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()

plt.savefig("outputs/loss_graph.png")
plt.show()
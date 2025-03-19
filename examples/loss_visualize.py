import json
import matplotlib.pyplot as plt

# Load the loss data from the JSON file
with open("/home/tq877/Tianyu/player_selection/MCP/examples/training_losses_bs_2 _ep_300 _lr_0.001 _sd_1.json", "r") as f:
    data = json.load(f)

# Initialize lists to store epochs and corresponding loss values
epochs = []
losses = []

# Each item in the data is a dictionary with one key-value pair (epoch: loss)
for item in data:
    for epoch_str, loss in item.items():
        epochs.append(int(epoch_str))
        losses.append(loss)

# Optionally, sort the epochs and losses (in case they're not already in order)
sorted_epochs, sorted_losses = zip(*sorted(zip(epochs, losses), key=lambda pair: pair[0]))

# Create a plot for the loss trend
plt.figure(figsize=(10, 6))
plt.plot(sorted_epochs, sorted_losses, marker='o')
plt.title("Training Loss Trend")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Save the plot to a file
plt.savefig("loss_trend_plot.png")

# Remove plt.show() if you don't want to display the plot
# plt.show()

import matplotlib.pyplot as plt
import re
import sys

def parse_log_file(filename):
    epochs = []
    train_losses = []
    val_losses = []

    pattern = re.compile(r"Epoch: (\d+), training loss: ([\d\.]+), validation loss: ([\d\.]+)")

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))

                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    return epochs, train_losses, val_losses

def plot_losses(epochs, train_losses, val_losses, run_name="run1"):
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Style settings
    plt.grid(True, alpha=0.3)
    
    # Plotting
    plt.plot(epochs, train_losses, label='Training Loss', color='#1f77b4', linestyle='-', linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=2, marker='o', markersize=5)
    
    # Labels and Title
    plt.title(f"Training and Validation Loss - {run_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    
    # Output
    output_filename = f"{run_name}_loss_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    log_file = "run1.txt"
    run_name = "run1"
    
    try:
        epochs, train_losses, val_losses = parse_log_file(log_file)
        if not epochs:
            print("No matching data found in the log file.")
        else:
            plot_losses(epochs, train_losses, val_losses, run_name)
    except FileNotFoundError:
        print(f"Error: File {log_file} not found.")

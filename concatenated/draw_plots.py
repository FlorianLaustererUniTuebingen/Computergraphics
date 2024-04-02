import matplotlib.pyplot as plt

# Initialize lists to hold epochs and accuracies
epochs = []
accuracies = []

# Open the file and read the lines
with open('./out/job.977909.out', 'r') as file:
    lines = file.readlines()

# Loop over the lines
for line in lines:
    # Check if the line contains 'Accuracy'
    if 'Accuracy' in line:
        # Split the line into words
        words = line.split()
        # The accuracy is the second word, remove the colon and convert to float
        accuracy = float(words[1].replace(':', ''))
        # Add the accuracy to the list
        accuracies.append(accuracy)
    # Check if the line contains 'Epoch'
    elif 'Epoch' in line:
        # Split the line into words
        words = line.split()
        # The epoch is the second word, remove the slash and convert to int
        epoch = int(words[1].split('/')[0])
        # Add the epoch to the list
        epochs.append(epoch)

# Create the plot
plt.plot(epochs, accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.savefig('accuracy_per_epoch.png')

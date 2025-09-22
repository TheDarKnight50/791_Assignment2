# download_data.py
from torchvision import datasets

print("Attempting to download MNIST dataset...")
# Download the training set
datasets.MNIST(root='./data', train=True, download=True)
# Download the test set
datasets.MNIST(root='./data', train=False, download=True)
print("✅ Download complete.")
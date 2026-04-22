import kagglehub

# Download latest version
path = kagglehub.dataset_download("Data/skin_dataset/ssl")

print("Path to dataset files:", path)
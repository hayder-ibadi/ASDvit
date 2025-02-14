# Path configurations

model_str = "google-vit-base-patch16-224-in21k"  
data_dir = 'Dataset'  # Path to the dataset
model_name = "Saved Model Folder"
num_train_epochs = 3
learning_rate = 5e-6
batch_size = 32  # Per device batch size
weight_decay = 0.02
warmup_steps = 50
test_size = 0.05  # Test set size
random_state = 83 # Random state for oversampling


# Define paths for saving models and logs
output_dir = model_name
logging_dir = './logs'
best_model_path = os.path.join(model_name, "best_model")

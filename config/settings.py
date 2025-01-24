# Path configurations
DATASET_PATH = "/home/necolas/Desktop/projects/dataset/AutismUpdated2/"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
OUTPUT_DIR = "ViT16B-SE-Block-v2"

# Training parameters
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 150,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 8,
    "learning_rate": 5e-6,
    "weight_decay": 0.02,
    "warmup_steps": 50,
    "logging_dir": "./logs",
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "save_total_limit": 1,
    "remove_unused_columns": False,
    "report_to": "none"
}

# Dataset parameters
CLASS_LABELS = ['autistic', 'non_autistic']
TEST_SIZE = 0.05
RANDOM_STATE = 83
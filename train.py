import warnings
import os
import gc
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    DefaultDataCollator,
    pipeline
)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config.settings import model_str, num_train_epochs, learning_rate, batch_size, weight_decay, warmup_steps, output_dir, logging_dir, best_model_path # Import from settings
from data.dataset_loader import load_and_preprocess_data
from models.model import ViTForImageClassificationWithSEBlock
from utils.transforms import create_transforms
from utils.metrics import compute_metrics, calculate_and_print_metrics, calculate_other_metrics

# Load and preprocess the dataset
dataset, labels_list = load_and_preprocess_data(data_dir = 'Dataset DIR')
train_data = dataset['train']
test_data = dataset['test']

processor = ViTImageProcessor.from_pretrained(model_str)
train_transforms, val_transforms = create_transforms(processor)

train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

model = ViTForImageClassificationWithSEBlock.from_pretrained(model_str, num_labels=len(labels_list))
model.config.id2label = dataset['train'].features['label'].int2str
model.config.label2id = dataset['train'].features['label'].str2int
print(model.num_parameters(only_trainable=True) / 1e6)


args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=8,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    remove_unused_columns=False,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none",
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.evaluate()

# Store metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_accuracy = 0.0

for epoch in range(num_train_epochs):
    print(f"Epoch {epoch+1}/{num_train_epochs}")

    train_result = trainer.train()
    train_metrics = train_result.metrics

    train_loss = train_metrics.get('loss', train_metrics.get('train_loss'))
    if train_loss is None:
        print("Warning: Training loss not found in metrics. Setting to None.")

    train_losses.append(train_loss)
    try:
        train_predictions = trainer.predict(train_data).predictions
        train_labels = [example['label'] for example in train_data] # Correct way to get labels
        train_predicted_labels = np.argmax(train_predictions, axis=1)
        train_accuracy = accuracy_score(train_labels, train_predicted_labels)
        train_accuracies.append(train_accuracy)
    except KeyError as e:
        print(f"KeyError while computing training accuracy: {e}")
        train_accuracy = None
        train_accuracies.append(None)


    print(f"  Training loss: {train_loss}")
    print(f"  Training accuracy: {train_accuracy}")

    eval_results = trainer.evaluate()
    val_loss = eval_results['eval_loss']
    val_accuracy = eval_results['eval_accuracy']

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"  Validation loss: {val_loss}")
    print(f"  Validation accuracy: {val_accuracy}")

    print(f"  Epoch {epoch+1} Metrics:")
    if train_loss is not None:
        print(f"    Training Loss: {train_loss:.4f}")
    else:
        print("    Training Loss: N/A")
    if train_accuracy is not None:
        print(f"    Training Accuracy: {train_accuracy:.4f}")
    else:
        print("    Training Accuracy: N/A")
    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation Accuracy: {val_accuracy:.4f}")

    trainer.save_model()

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        trainer.save_model(best_model_path)
        print(f"  Saving best model with accuracy: {best_accuracy:.4f}")

epochs = range(1, num_train_epochs + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r', label='Training Loss', marker="")
plt.plot(epochs, val_losses, 'b', label='Validation Loss', marker="")
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'r', label='Training Accuracy', marker="")
plt.plot(epochs, val_accuracies, 'b', label='Validation Accuracy', marker="")
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")  # Save the plot as an image
plt.show()

print(f"Best model saved to {best_model_path} with accuracy {best_accuracy:.4f}")

outputs = trainer.predict(test_data)
print(outputs.metrics)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

calculate_and_print_metrics(y_true, y_pred, labels_list)
calculate_other_metrics(y_true, y_pred, outputs)

trainer.save_model()

pipe = pipeline('image-classification', model=output_dir, device=0) # Use output_dir, not model_name

image = test_data[1]["image"]
print(pipe(image))
print(dataset['train'].features['label'].int2str(test_data[1]["label"]))

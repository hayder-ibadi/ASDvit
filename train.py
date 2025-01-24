from transformers import ViTImageProcessor, TrainingArguments, Trainer
import evaluate
from datasets import load_dataset
import torch
from config.settings import *
from data.dataset_loader import load_dataset, balance_data, create_hf_dataset
from models.model import ViTWithSE
from utils.transforms import get_transforms
from utils.metrics import plot_confusion_matrix, print_metrics

def main():
    # Load and prepare data
    df = load_dataset(DATASET_PATH)
    balanced_df = balance_data(df)
    dataset = create_hf_dataset(balanced_df, CLASS_LABELS)

    # Initialize processor and model
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTWithSE.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_LABELS),
        id2label={i: l for i, l in enumerate(CLASS_LABELS)},
        label2id={l: i for i, l in enumerate(CLASS_LABELS)}
    )

    # Apply transforms
    transforms = get_transforms(processor)
    dataset['train'].set_transform(lambda x: {
        'pixel_values': transforms['train'](x['image'].convert('RGB')),
        'label': x['label']
    })
    dataset['test'].set_transform(lambda x: {
        'pixel_values': transforms['val'](x['image'].convert('RGB')),
        'label': x['label']
    })

    # Training setup
    training_args = TrainingArguments(**TRAINING_ARGS)
    
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return accuracy.compute(predictions=predictions.argmax(axis=1), references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=processor,
    )

    # Training
    trainer.train()
    trainer.save_model()

    # Evaluation
    results = trainer.predict(dataset['test'])
    probs = results.predictions
    preds = probs.argmax(axis=1)
    
    plot_confusion_matrix(results.label_ids, preds, CLASS_LABELS)
    print_metrics(results.label_ids, preds, probs, CLASS_LABELS)

if __name__ == "__main__":
    main()
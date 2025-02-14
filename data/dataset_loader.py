import pandas as pd
from datasets import Dataset, Image, ClassLabel
from pathlib import Path
import gc
from imblearn.over_sampling import RandomOverSampler
from config.settings import data_dir, test_size, random_state  # Import from settings

def load_and_preprocess_data(data_dir):
    
    file_names = []
    labels = []

    for file in sorted((Path(data_dir).glob('*/*/*.*'))):
        label = str(file).split('/')[-2]
        labels.append(label)
        file_names.append(str(file))

    print(len(file_names), len(labels))

    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
    print(df.shape)

    print(df['label'].unique())

    # Oversample to balance the dataset
    y = df[['label']]
    df = df.drop(['label'], axis=1)
    ros = RandomOverSampler(random_state=random_state)  
    df, y_resampled = ros.fit_resample(df, y)
    del y
    df['label'] = y_resampled
    del y_resampled
    gc.collect()

    print(df.shape)

    dataset = Dataset.from_pandas(df).cast_column("image", Image())

    labels_list = ['autistic', 'non_autistic']
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels_list):
        label2id[label] = i
        id2label[i] = label

    print("Mapping of IDs to Labels:", id2label, '\n')
    print("Mapping of Labels to IDs:", label2id)

    # Add label mappings to the dataset object
    dataset.label2id = label2id  # Assign to the dataset object
    dataset.id2label = id2label # Assign to the dataset object


    ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

    def map_label2id(example):
        example['label'] = ClassLabels.str2int(example['label'])
        return example

    dataset = dataset.map(map_label2id, batched=True)
    dataset = dataset.cast_column('label', ClassLabels)
    dataset = dataset.train_test_split(test_size=test_size, shuffle=True, stratify_by_column="label") # Use test_size from config

    return dataset, labels_list

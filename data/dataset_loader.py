import pandas as pd
from pathlib import Path
from datasets import Dataset, Image, ClassLabel
from imblearn.over_sampling import RandomOverSampler
import gc

def load_dataset(dataset_path):
    file_names = []
    labels = []
    
    for file in sorted(Path(dataset_path).glob('*/*/*.*')):
        label = str(file).split('/')[-2]
        labels.append(label)
        file_names.append(str(file))
    
    return pd.DataFrame({"image": file_names, "label": labels})

def balance_data(df):
    y = df[['label']]
    df = df.drop(['label'], axis=1)
    ros = RandomOverSampler(random_state=83)
    df, y_resampled = ros.fit_resample(df, y)
    df['label'] = y_resampled
    del y, y_resampled
    gc.collect()
    return df

def create_hf_dataset(df, class_labels):
    dataset = Dataset.from_pandas(df).cast_column("image", Image())
    class_labels = ClassLabel(names=class_labels)
    dataset = dataset.map(lambda ex: {'label': class_labels.str2int(ex['label'])}, batched=True)
    return dataset.train_test_split(test_size=0.05, stratify_by_column="label")
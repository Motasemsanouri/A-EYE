import os
import shutil
import zipfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv2D, GlobalAveragePooling2D,
                                     Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (LearningRateScheduler, EarlyStopping,
                                        ModelCheckpoint, Callback)

import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {zip_path} → {extract_to}")


def analyze_distribution(dataset_path):
    label_counts = {
        label.name: len(list(label.glob('*')))
        for label in Path(dataset_path).iterdir() if label.is_dir()
    }
    df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Label', y='Count', palette='viridis')
    plt.title('Image Distribution per Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return df


def split_dataset(df, labels, test_size=0.2):
    df = df[df['labels'].isin(labels)]
    X_train, X_test, y_train, y_test = train_test_split(df['path'], df['labels'], test_size=test_size, random_state=42)
    train_df = pd.DataFrame({'path': X_train, 'labels': y_train, 'set': 'train'})
    test_df = pd.DataFrame({'path': X_test, 'labels': y_test, 'set': 'test'})
    return train_df, test_df


def organize_files(df, target_dir):
    for _, row in df.iterrows():
        dest = Path(target_dir) / row['set'] / row['labels']
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['path'], dest / Path(row['path']).name)
    print(f"Files moved to {target_dir}")


def build_generators(train_dir, test_dir, img_size=(224, 224), batch_size=32):
    train_gen = ImageDataGenerator(#provide real time data auggumetation for image dataset
        rescale=1. / 255, rotation_range=10, horizontal_flip=True,
        zoom_range=0.1, brightness_range=[0.8, 1.2], fill_mode='nearest'
    ).flow_from_directory(train_dir, target_size=img_size, class_mode='categorical',
                          batch_size=batch_size, shuffle=True)

    test_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_dir, target_size=img_size, class_mode='categorical',
        batch_size=batch_size, shuffle=False)

    return train_gen, test_gen


def create_model(input_shape=(224, 224, 3), num_classes=4):
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = True
    model = Sequential([
        base,
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model


def lr_scheduler(epoch, lr):
    base_lr, max_lr = 1e-5, 1e-3
    step_size = 2000
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))


class StopOnAccuracy(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= 0.93 and logs.get('val_loss') < 0.13:
            print('Stopping early: Accuracy > 93% and Loss < 0.13')#stop training when trainig performence improved
            self.model.stop_training = True


def compile_and_train(model, train_gen, val_gen, class_weights, save_path):
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])#adam optimizer an 
                                                                                                          #algorithem for updating

    callbacks = [
        StopOnAccuracy(),
        LearningRateScheduler(lr_scheduler),
        EarlyStopping(patience=30, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(train_gen, epochs=150, validation_data=val_gen,
                        class_weight=class_weights, callbacks=callbacks)
    return history


def plot_metrics(history):
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Val')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def export_model(model, export_path='./saved_model_1'):
    tf.saved_model.save(model, f"{export_path}/eye_diseases_model")
    converter = tf.lite.TFLiteConverter.from_saved_model(f"{export_path}/eye_diseases_model")
    tflite_model = converter.convert()
    os.makedirs(f"{export_path}/tflite_model", exist_ok=True)

    with open(f"{export_path}/tflite_model/eye_diseases_model.tflite", 'wb') as f:
        f.write(tflite_model)

    print("Model exported as SavedModel and TFLite format.")



if __name__ == "__main__":
    zip_file = 'archive (2).zip'
    extract_folder = Path('extracted_data')
    dataset_folder = extract_folder / 'archive' / 'dataset'
    desired_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

    extract_zip(zip_file, extract_folder)
    df = analyze_distribution(dataset_folder)

    image_data = []
    for label in desired_labels:
        files = list((dataset_folder / label).glob('*'))
        image_data += [{'path': str(f), 'labels': label} for f in files]
    df_full = pd.DataFrame(image_data)

    train_df, test_df = split_dataset(df_full, desired_labels)
    organize_files(pd.concat([train_df, test_df]), 'cleaned_dataset')

    train_gen, test_gen = build_generators('cleaned_dataset/train', 'cleaned_dataset/test')
    model = create_model()

    class_totals = train_df['labels'].value_counts().to_dict()
    total = sum(class_totals.values())
    class_weights = {i: total / (len(desired_labels) * count) for i, (_, count) in enumerate(class_totals.items())}

    history = compile_and_train(model, train_gen, test_gen, class_weights, 'best_model.keras')
    plot_metrics(history)
    export_model(model)

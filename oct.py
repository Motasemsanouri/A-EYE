import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, GlobalAveragePooling2D,
                                     Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (LearningRateScheduler, EarlyStopping,
                                        ModelCheckpoint, Callback)

import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

def analyze_distribution(dataset_path):
    label_counts = {
        label.name: len(list(label.glob('*')))
        for label in Path(dataset_path).iterdir() if label.is_dir() and label.name in ['CNV', 'DME', 'DRUSEN', 'NORMAL']
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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(df['path'], df['labels'], test_size=test_size, random_state=42)
    train_df = pd.DataFrame({'path': X_train, 'labels': y_train, 'set': 'train'})
    test_df = pd.DataFrame({'path': X_test, 'labels': y_test, 'set': 'test'})
    return train_df, test_df

def organize_files(df, target_dir):
    for _, row in df.iterrows():
        if row['labels'] not in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
            print(f"⚠️ Skipping unknown label: {row['labels']}")
            continue
        dest = Path(target_dir) / row['set'] / row['labels']
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['path'], dest / Path(row['path']).name)
    print(f"Files moved to {target_dir}")

def build_generators(train_dir, test_dir, img_size=(224, 224), batch_size=32):
    train_gen = ImageDataGenerator(
        rescale=1. / 255, rotation_range=15, horizontal_flip=True,
        zoom_range=0.15, brightness_range=[0.7, 1.3], fill_mode='nearest',
        shear_range=0.1, width_shift_range=0.1, height_shift_range=0.1
    ).flow_from_directory(train_dir, target_size=img_size, class_mode='categorical',
                          batch_size=batch_size, shuffle=True)

    test_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_dir, target_size=img_size, class_mode='categorical',
        batch_size=batch_size, shuffle=False)

    return train_gen, test_gen

def create_model(input_shape=(224, 224, 3), num_classes=4):
    base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None)
    base.trainable = True
    model = Sequential([
        base,
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def lr_scheduler(epoch, lr):
    return 1e-4

class StopOnAccuracy(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= 0.93 and logs.get('val_loss') < 0.13:
            print('Stopping early: Accuracy > 93% and Loss < 0.13')
            self.model.stop_training = True

def compile_and_train(model, train_gen, val_gen, class_weights, save_path):
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        StopOnAccuracy(),
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(filepath=str(save_path), monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(train_gen, epochs=100, validation_data=val_gen,
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

def evaluate_model(model, test_gen):
    test_gen.reset()
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def export_model(model, export_path='./oct_model_export'):
    tf.saved_model.save(model, f"{export_path}/oct_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model(f"{export_path}/oct_saved_model")
    tflite_model = converter.convert()
    os.makedirs(f"{export_path}/tflite_model", exist_ok=True)

    with open(f"{export_path}/tflite_model/oct_model.tflite", 'wb') as f:
        f.write(tflite_model)

    model.save(f"{export_path}/oct_model.h5")

    print("Model exported as H5 and TFLite format.")

if __name__ == "__main__":
    dataset_folder = Path('C:/Users/ASUS/Desktop/graduation_project/balanced OCT')
    desired_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    df = analyze_distribution(dataset_folder)

    image_data = []
    for label in desired_labels:
        label_dir = dataset_folder / label
        if label_dir.exists():
            files = list(label_dir.glob('*'))
            image_data += [{'path': str(f), 'labels': label} for f in files]
    df_full = pd.DataFrame(image_data)

    train_df, test_df = split_dataset(df_full, desired_labels)
    organize_files(pd.concat([train_df, test_df]), 'cleaned_dataset')

    train_gen, test_gen = build_generators('cleaned_dataset/train', 'cleaned_dataset/test')
    model = create_model()

    class_totals = train_df['labels'].value_counts().to_dict()
    total = sum(class_totals.values())
    label_to_index = train_gen.class_indices
    class_weights = {
        label_to_index[label]: total / (len(desired_labels) * count)
        for label, count in class_totals.items() if label in label_to_index
    }

    history = compile_and_train(model, train_gen, test_gen, class_weights, 'oct_model.h5')
    plot_metrics(history)
    evaluate_model(model, test_gen)
    export_model(model)

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tf_keras.models import Model
from tf_keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tf_keras.applications.inception_v3 import InceptionV3
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tf_keras.optimizers import Adam
from preprocessing import preprocessing_function

wandb.init(project="drowsiness-detection", config={
    "learning_rate_phase1": 1e-3,
    "learning_rate_phase2": 1e-5,
    "epochs_phase1": 5,  
    "epochs_phase2": 15,   
    "batch_size": 32,
    "architecture": "InceptionV3_TwoPhase",
    "dataset": "Drowsiness",
})
config = wandb.config

CLASSES = 2
    
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions = Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
      
model.compile(optimizer=Adam(learning_rate=config.learning_rate_phase1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

WIDTH = 299
HEIGHT = 299
BATCH_SIZE = config.batch_size
TRAIN_DIR = 'dataset/train'

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_function,
    rotation_range=15,         
    width_shift_range=0.1,      
    height_shift_range=0.1,     
    shear_range=0.15,           
    zoom_range=0.15,            
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)   

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')       

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')     

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
    WandbMetricsLogger(),
    WandbModelCheckpoint(filepath="model.keras", monitor="val_loss", save_best_only=True)
]

print("\n--- STARTING PHASE 1: WARM-UP ---")
history_phase1 = model.fit(
    train_generator,
    epochs=config.epochs_phase1,
    validation_data=validation_generator,
    callbacks=callbacks)

print("\n--- STARTING PHASE 2: FINE-TUNING ---")
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=config.learning_rate_phase2),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_phase2 = model.fit(
    train_generator,
    epochs=config.epochs_phase2,
    validation_data=validation_generator,
    callbacks=callbacks)
  
model.save("model.keras")
wandb.finish()
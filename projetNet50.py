import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.image import imread
from skimage.color import gray2rgb
import seaborn as sns

# ------------------------------
# 1️⃣ Chemins
# ------------------------------
my_data_dir = "dataset/classification"
train_path = os.path.join(my_data_dir, "train")
test_path = os.path.join(my_data_dir, "test")

# ------------------------------
# 1a️⃣ Exploration des classes
# ------------------------------
for cls in ["glioma", "meningioma", "no_tumor", "pituitary"]:
    print(f"Train - {cls}: {len(os.listdir(os.path.join(train_path, cls)))} images")
    print(f"Test  - {cls}: {len(os.listdir(os.path.join(test_path, cls)))} images")

# ------------------------------
# 1b️⃣ Vérification dimensions images
# ------------------------------
dim1, dim2 = [], []

for image_filename in os.listdir(os.path.join(train_path, "glioma")):
    img = imread(os.path.join(train_path, "glioma", image_filename))
    if len(img.shape) == 2:
        img = gray2rgb(img)
    d1, d2, _ = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1, y=dim2)
plt.show()
print("Dimension moyenne:", np.mean(dim1), np.mean(dim2))
print("Pixel min:", np.min(img), "Pixel max:", np.max(img))

# ------------------------------
# 2️⃣ Paramètres
# ------------------------------
image_shape = (224, 224, 3)
batch_size = 32
num_classes = 4
validation_split = 0.1
AUTOTUNE = tf.data.AUTOTUNE

# ------------------------------
# 3️⃣ Dataset tf.data
# ------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=image_shape[:2],
    batch_size=batch_size,
    label_mode='categorical',
    validation_split=validation_split,
    subset="training",
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=image_shape[:2],
    batch_size=batch_size,
    label_mode='categorical',
    validation_split=validation_split,
    subset="validation",
    seed=42
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=image_shape[:2],
    batch_size=batch_size,
    label_mode='categorical'
)

# ------------------------------
# 4️⃣ Normalisation et augmentation
# ------------------------------
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomTranslation(0.1, 0.1)
])

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------
# 5️⃣ Modèle Sequential avec ResNet50
# ------------------------------
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=image_shape)
base_model.trainable = False

model = Sequential([
    data_augmentation,
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# 6️⃣ EarlyStopping
# ------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ------------------------------
# 7️⃣ Fonction de détection overfitting/underfitting
# ------------------------------
def detect_fit(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    last_acc, last_val_acc = acc[-1], val_acc[-1]

    print("\n📊 Analyse du modèle :")
    if last_acc > 0.95 and last_val_acc < 0.80:
        print("⚠️ Overfitting détecté")
    elif last_acc < 0.70 and last_val_acc < 0.70:
        print("⚠️ Underfitting détecté")
    else:
        print("✅ Pas d’overfitting ni underfitting majeur")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend(); plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend(); plt.title("Loss")
    plt.show()

# ------------------------------
# 8️⃣ Entraînement head
# ------------------------------
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])
detect_fit(history)

# ------------------------------
# 9️⃣ Fine-tuning
# ------------------------------
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])
detect_fit(history_finetune)

# ------------------------------
# 🔟 Évaluation et matrice de confusion
# ------------------------------
y_true, y_pred = [], []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
class_names = test_ds.class_names if hasattr(test_ds, 'class_names') else [str(i) for i in range(num_classes)]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de Confusion")
plt.show()

# ------------------------------
# 1️⃣1️⃣ Sauvegarde modèle
# ------------------------------
model.save("projetResnet50.keras")

# -*- coding: utf-8 -*-
"""
Highest-accuracy training script for two-character classification (miku vs knd).
Features:
- stronger data augmentation (shear, brightness, etc.)
- class weights to handle imbalance
- transfer learning with MobileNetV2: freeze -> train head -> unfreeze last 30 layers -> fine-tune
- callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Test Time Augmentation (TTA) for more robust evaluation
- saves best model and final model
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# -------------------- 配置 --------------------
dataset_path = r"C:\Users\User\Downloads\UI\GA_two_character\dataset"
img_width, img_height = 128, 128
test_size = 0.2
batch_size = 16
initial_epochs = 12       # 先训练 top 层的 epoch
fine_tune_epochs = 20     # 解冻后微调的 epoch
total_epochs = initial_epochs + fine_tune_epochs
seed = 42
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, "best_two_charate.h5")
final_model_path = os.path.join(model_dir, "final_two_charate.h5")
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# -------------------- 角色 --------------------
class_names = ["Hatsune Miku", "Yoisaki Kanade"]
num_classes = len(class_names)
print("Classes:", class_names)

# -------------------- 读取数据 --------------------
images, labels = [], []
img_extensions = ('.jpg', '.jpeg', '.png', '.webp')

for idx, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.exists(class_folder):
        raise FileNotFoundError(f"Class folder not found: {class_folder}")
    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith(img_extensions):
            img_path = os.path.join(class_folder, img_file)
            try:
                img = load_img(img_path, target_size=(img_width, img_height))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Warning: failed to load {img_path}: {e}")

images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels, dtype='int32')
labels_cat = to_categorical(labels, num_classes)

# -------------------- 划分数据集 --------------------
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_cat, test_size=test_size, random_state=seed, stratify=labels
)

# 统计样本数
print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Class distribution (train):", np.bincount(np.argmax(y_train, axis=1)))

# -------------------- 数据增强（训练 & 验证） --------------------
# 更强的增强设置（适合动漫头像）
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.12,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode='nearest'
)

# 验证集只做最小处理（确保评估稳定）
val_datagen = ImageDataGenerator()

train_datagen.fit(X_train)

# 使用 flow 而非 flow_from_directory（我们已加载为数组）
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
val_generator = val_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

# -------------------- 计算类别权重 --------------------
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -------------------- 构建模型（MobileNetV2 base + head） --------------------
base_model = MobileNetV2(weights='imagenet',
                         include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=5e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------- callbacks --------------------
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

callbacks = [early_stop, checkpoint, reduce_lr]

# -------------------- 阶段一：训练 head（冻结 base） --------------------
history_head = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# -------------------- 解冻后半部分进行微调 --------------------
# 解冻 base_model 的最后 30 层（或根据模型深度选择）
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 用更小的学习率微调所有可训练层
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history_finetune = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# 合并历史（便于绘图）
def merge_history(h1, h2):
    merged = {}
    for k in h1.history.keys():
        merged[k] = h1.history[k] + h2.history.get(k, [])
    for k in h2.history.keys():
        if k not in merged:
            merged[k] = h2.history[k]
    return merged

merged_history = merge_history(history_head, history_finetune)

# -------------------- 保存最终模型 --------------------
model.save(final_model_path)
print("Final model saved to:", final_model_path)

# -------------------- 可视化训练曲线 --------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(merged_history['accuracy'], label='Train Accuracy')
plt.plot(merged_history.get('val_accuracy', []), label='Validation Accuracy')
plt.title('Train / Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(merged_history['loss'], label='Train Loss')
plt.plot(merged_history.get('val_loss', []), label='Validation Loss')
plt.title('Train / Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- 评估：常规预测 --------------------
# 加载最佳模型（由 ModelCheckpoint 保存）
best_model = load_model(best_model_path)
y_test_labels = np.argmax(y_test, axis=1)

y_pred_prob = best_model.predict(X_test, batch_size=batch_size, verbose=1)
y_pred_labels = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test_labels, y_pred_labels)
print(f"Test Accuracy (no TTA): {acc:.4f}")

cm = confusion_matrix(y_test_labels, y_pred_labels)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=class_names))

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (no TTA)")
plt.show()

# -------------------- Test Time Augmentation (TTA) --------------------
# 对每张测试样本做多次增强并平均预测概率
def tta_predict(model, X, tta_steps=10):
    probs = np.zeros((X.shape[0], num_classes), dtype=np.float32)
    tta_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )
    for i in range(tta_steps):
        augmented = tta_datagen.flow(X, batch_size=batch_size, shuffle=False).next()
        aug_all = []
        for j in range(X.shape[0]):
            x_aug = tta_datagen.random_transform(X[j])
            aug_all.append(x_aug)
        aug_all = np.array(aug_all)
        p = model.predict(aug_all, batch_size=batch_size, verbose=0)
        probs += p
    probs /= tta_steps
    return probs

print("Running TTA predictions (this may take a while)...")
tta_steps = 8
y_pred_prob_tta = tta_predict(best_model, X_test, tta_steps=tta_steps)
y_pred_labels_tta = np.argmax(y_pred_prob_tta, axis=1)
acc_tta = accuracy_score(y_test_labels, y_pred_labels_tta)
print(f"Test Accuracy (TTA {tta_steps}): {acc_tta:.4f}")

cm_tta = confusion_matrix(y_test_labels, y_pred_labels_tta)
print("Confusion Matrix (TTA):")
print(cm_tta)
print("Classification Report (TTA):")
print(classification_report(y_test_labels, y_pred_labels_tta, target_names=class_names))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_tta, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (TTA {tta_steps})")
plt.show()

# -------------------- 打印总结 --------------------
print("Training complete.")
print(f"Best model path: {best_model_path}")
print(f"Final model path: {final_model_path}")

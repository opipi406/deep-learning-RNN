# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    return fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
fashion_mnist = keras.datasets.fashion_mnist

# 学習データ60000 28x28, テストデータ10000 28x28
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
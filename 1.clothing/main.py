# import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """ mnistから画像データを取得する

    :return: (学習データ画像, 学習データラベル), (テストデータ画像, テストデータラベル) のndArrayを持つ2種類のタプルを返却
    """
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

def _plot_test():
    plt.figure(figsize=(8, 8))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

def plot_image(i, predictions_array, true_labels, images):
    """ 衣類画像の描画
    :param i: index of datas
    :param predictions_array: 予測値の配列
    :param true_labels: 正解データのラベル配列
    :param images: 正解データの画像配列
    :return:
    """
    predictions_array, true_labels, images = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(images, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_labels:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_labels]),
        color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def plot_result(i):
    """ 予測結果の統計の出力

    テスト画像、予測されたラベル、正解ラベルを表示します。
    正しい予測は青で、間違った予測は赤で表示しています。

    :param i: index of datas
    :return:
    """
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

def plot_results(num_rows=5, num_cols=3):
    """ 予測結果の統計の出力予測結果の統計の出力

    テスト画像、予測されたラベル、正解ラベルを表示します。
    正しい予測は青で、間違った予測は赤で表示しています。

    :param num_rows:
    :param num_cols:
    :return:
    """
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 学習データ60000 28x28, テストデータ10000 28x28
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# draw_images()

#=================================================================
#   モデルの構築
#=================================================================
# 層の設定
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
model.fit(train_images, train_labels, epochs=5)

# 損失値, 正答率
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print()
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# 衣類画像の予測
predictions = model.predict(test_images)
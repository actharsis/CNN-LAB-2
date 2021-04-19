# Решение задачи классификации изображений из набора данных Food-101 с использованием нейронных сетей глубокого обучения и техники обучения Transfer Learning
## Нейронная сеть EfficientNet-B0 со случайным начальным приближением
Файл:
```
CNN-food-101-master/train.py
```
Структура сети:
```python
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
outputs = tf.keras.applications.EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES, input_tensor=inputs)
return tf.keras.Model(inputs=inputs, outputs=outputs.output)
```
Метрика качества:

![Legend 1](https://user-images.githubusercontent.com/24518594/115115815-20923000-9f9f-11eb-86e5-ad1c3c7fa727.png)
![graph1](https://github.com/actharsis/lab2/blob/main/graphs/epoch_categorical_accuracy_1.svg)
Функция потерь:
![graph2](https://github.com/actharsis/lab2/blob/main/graphs/epoch_loss_1.svg)
## EfficientNet-B0 в сочетании с техникой обучения Transfer Learning
Файл:
```
CNN-food-101-master/transfer_train.py
```
Структура сети:
```python
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
model = EfficientNetB0(include_top=False, weights="imagenet", classes=NUM_CLASSES, input_tensor=inputs)
model.trainable = False
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
return tf.keras.Model(inputs=inputs, outputs=outputs)
```
Метрика качества:

![Legend 2](https://user-images.githubusercontent.com/24518594/115115927-b037de80-9f9f-11eb-9a5d-efa5721918cc.png)
![graph3](https://github.com/actharsis/lab2/blob/main/graphs/epoch_categorical_accuracy_2.svg)
Функция потерь:
![graph4](https://github.com/actharsis/lab2/blob/main/graphs/epoch_loss_2.svg)
## Анализ результатов
Первая попытка обучения нейросети оказалась неудачной: максимальная точность по метрике качества ~35%. Для второй попытки была использована техника Transfer Learning, в частности были использованы предобученные веса. В результате сеть показала большую точность - 84% на train и 65% на validation. По графикам можно сказать, что нейросеть переобучается и обучение следовало бы остановить уже к 5-ой эпохе. Как плюс техники Transfer Learning можно выделить увеличение точности(в нашем случае +50% на train и +30% на validation).

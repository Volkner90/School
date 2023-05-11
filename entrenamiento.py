import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Preparación de los datos
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/entrenamiento',
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.1,
    subset='training'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/validacion',
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.1,
    subset='validation'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/prueba', 
    seed=123,
    image_size=(128, 128),
    batch_size=1,
    shuffle=False
)

class_names = train_ds.class_names

# Definición del modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Normalization())

model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), name= 'conv2d_1',padding='same', data_format='channels_last', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), name='conv2d_2',padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), name='conv2d_3',padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), name='conv2d_4',padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), name='conv2d_5',padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(units=300, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

tf.random.set_seed(1)
model.build(input_shape=(None,128,128,3))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())

history = model.fit(train_ds, validation_data=val_ds, epochs=3, shuffle=True)

print(model.evaluate(train_ds))
print(model.evaluate(val_ds))
print(model.evaluate(test_ds))

epochs=1

# Realizando predicciones
y_pred = model.predict(test_ds)
predicate_classes = np.argmax(y_pred, axis=1)

y_test = []
for image, label in test_ds:
    for i in label:
        y_test.append(i.numpy())

# Métricas
print(metrics.classification_report(y_test, predicate_classes))

# Heatmap
cf = confusion_matrix(y_test, predicate_classes)
sns.heatmap(cf, annot=True)
print(class_names)
plt.show()
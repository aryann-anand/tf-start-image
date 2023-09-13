import tensorflow as tf
print("tensorflow version:" , tf.__version__)

# loading the mnist dataset
mnist = tf.keras.datasets.mnist

# preparing the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# build a tf.keras.Sequantial Model:

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fxn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fxn(y_train[:1], predictions).numpy()

'''
set the optimizer class to 'adam'
set the loss to loss_fxn function defined earlier
specify a metric to be evaluated for the model by setting (metrics) parameter to (accuracy)
'''

model.compile(optimizer='adam',
              loss=loss_fxn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
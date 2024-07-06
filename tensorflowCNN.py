import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.utils import get_file
import numpy as np
import gzip

#Loads the Fashion MNIST dataset
def load_fashion_mnist():
    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=base_url + fname, cache_subdir='datasets'))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

#Load data
(x_train, y_train), (x_test, y_test) = load_fashion_mnist()

#Normalize pixel values to the range [0, 1]
x_train = x_train/255.0
x_test = x_test/255.0

#Transform data to the range [-1, 1]
x_train = 2 * x_train - 1
x_test = 2 * x_test - 1

#Add color channel
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

train_len = len(x_train)
test_len = len(x_test)
batch_size = 32
buffer_size = 10000

#Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).repeat()

#Create testing dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

#CNN Model
class tensorflow_CNN(Model):
    #Declare and assign layers in the network
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, (3,3), strides = 2, activation="relu")
        self.pool1 = MaxPooling2D((2, 2))
        self.drop1 = Dropout(0.5)
        self.conv2 = Conv2D(64, (3,3), strides = 2, activation="relu")
        self.pool2 = MaxPooling2D((2, 2))
        self.drop2 = Dropout(0.5)
        self.flatten = Flatten()
        self.d1 = Dense(32, activation="relu")
        self.d2 = Dense(10)

    #Process input into final logits
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

#Instantiate model
model = tensorflow_CNN()

#Define loss function and optimizer
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

#Training metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#Testing metrics
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#Runs one training step: updates weights according to gradient and records loss and accuracy
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

#Runs one testing step: does not update weights but records loss and accuracy on testing data
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_obj(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 5
steps_per_epoch = train_len // batch_size
validation_steps = test_len // batch_size

for epoch in range(EPOCHS):
    #Reset the metrics at the start of each epoch
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    #Training epoch
    for step in range(steps_per_epoch):
        imgs, labels = next(iter(train_dataset))
        train_step(imgs, labels)

    #Testing epoch
    for step in range(validation_steps):
        imgs, labels = next(iter(test_dataset))
        test_step(imgs, labels)

    #Print relevant info
    print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result():0.2f}, ' #Training loss for the epoch
    f'Accuracy: {train_accuracy.result() * 100:0.2f}, ' #Training accuracy for the epoch
    f'Test Loss: {test_loss.result():0.2f}, ' #Testing loss for the epoch
    f'Test Accuracy: {test_accuracy.result() * 100:0.2f}' #Testing accuracy for the epoch
    )

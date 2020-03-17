import tensorflow as tf

from nalp.models.adversarial.dcgan import DCGAN

# Defining a constant to hold the input noise dimension
N_NOISE = 100

# Defining a constant to hold the batch size
BATCH_SIZE = 256

# Loading the MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshaping the training data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

# Normalizing the data
x_train = (x_train - 127.5) / 127.5

# Creating the dataset from shuffle and batched data
dataset = tf.data.Dataset.from_tensor_slices(
    x_train).shuffle(100000).batch(BATCH_SIZE)

# Creating the DCGAN
dcgan = DCGAN(gen_input=N_NOISE, alpha=0.2, dropout=0.3)

# Compiling the DCGAN
dcgan.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
            loss=tf.losses.BinaryCrossentropy(from_logits=True))

# Fitting the DCGAN
dcgan.fit(dataset, epochs=1)

# Saving DCGAN weights
dcgan.save_weights('trained/dcgan', save_format='tf')

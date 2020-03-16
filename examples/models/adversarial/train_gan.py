import tensorflow as tf

from nalp.models.adversarial.gan import GAN

# Loading the MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshaping the training data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

# Normalizing the data to [-1, 1]
x_train = (x_train - 127.5) / 127.5

# Creating the dataset from shuffle and batched data
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(100000).batch(256)

# Creating the GAN
gan = GAN()

# Compiling the GAN
gan.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
            loss=tf.losses.BinaryCrossentropy(from_logits=True))

# Fitting the GAN
gan.fit(dataset, epochs=5)

# Creating a noise tensor for further sampling
z = tf.random.normal([1, 100])

# Sampling artificial data
gan.sample(z)

# Saving GAN weights
gan.save_weights('trained/gan', save_format='tf')

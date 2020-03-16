import tensorflow as tf
from tensorflow.keras import Model

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class AdversarialWrapper(Model):
    """An AdversarialWrapper class is responsible for customly implementing
    Generative Adversarial Networks.

    """

    def __init__(self, discriminator, generator, name=''):
        """Initialization method.

        Args:
            discriminator (Model): Network's discriminator architecture.
            generator (Model): Network's generator architecture.
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(AdversarialWrapper, self).__init__(name=name)

        # Defining the discriminator network
        self.D = discriminator

        # Defining the generator network
        self.G = generator

    def compile(self, optimizer, loss):
        """Main building method.

        Args:
            optimizer (tf.optimizers): An optimizer instance.
            loss (tf.loss): A loss instance.

        """

        # Creates an optimizer object for the discriminator
        self.D_optimizer = optimizer

        # Creates an optimizer object for the generator
        self.G_optimizer = optimizer

        # Defining the loss function
        self.loss = loss

        # Defining a loss metric for the discriminator
        self.D_loss = tf.metrics.Mean(name='D_loss')

        # Defining a loss metric for the generator
        self.G_loss = tf.metrics.Mean(name='G_loss')

    def discriminator_loss(self, y, y_fake):
        """Calculates the loss out of the discriminator architecture.

        Args:
            y (tf.Tensor): A tensor containing the real data targets.
            y_fake (tf.Tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the discriminator network.

        """

        # Calculates the real data loss
        real_loss = self.loss(tf.ones_like(y), y)

        # Calculates the fake data loss
        fake_loss = self.loss(tf.zeros_like(y_fake), y_fake)

        return real_loss + fake_loss

    def generator_loss(self, y_fake):
        """Calculates the loss out of the generator architecture.

        Args:
            y_fake (tf.Tensor): A tensor containing the fake data targets.

        Returns:
            The loss based on the generator network.

        """

        return self.loss(tf.ones_like(y_fake), y_fake)

    @tf.function
    def step(self, x):
        """Performs a single batch optimization step.

        Args:
            x (tf.Tensor): A tensor containing the inputs.

        """

        # Defines a random noise signal as the generator's input
        z = tf.random.normal([x.shape[0], 1, 1, self.G.n_input])

        # Using tensorflow's gradient
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # Generates new data, e.g., G(z)
            x_fake = self.G(z)

            # Samples fake targets from the discriminator, e.g., D(G(z))
            y_fake = self.D(x_fake)

            # Samples real targets from the discriminator, e.g., D(x)
            y = self.D(x)

            # Calculates the generator loss upon D(G(z))
            G_loss = self.generator_loss(y_fake)

            # Calculates the discriminator loss upon D(x) and D(G(z))
            D_loss = self.discriminator_loss(y, y_fake)

        # Calculate the gradients based on generator's loss for each training variable
        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)

        # Calculate the gradients based on discriminator's loss for each training variable
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        # Applies the generator's gradients using an optimizer
        self.G_optimizer.apply_gradients(
            zip(G_gradients, self.G.trainable_variables))

        # Applies the discriminator's gradients using an optimizer
        self.D_optimizer.apply_gradients(
            zip(D_gradients, self.D.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(G_loss)

        # Updates the discriminator's loss state
        self.D_loss.update_state(D_loss)

    def fit(self, batches, epochs=100):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')
            
            # Resetting states to further append losses
            self.G_loss.reset_states()
            self.D_loss.reset_states()

            # Iterate through all possible training batches
            for batch in batches:
                # Performs the optimization step
                self.step(batch)

            logger.info(
                f'Loss(G): {self.G_loss.result().numpy():.4f} | Loss(D): {self.D_loss.result().numpy():.4f}')

    @tf.function
    def sample(self, z):
        """Uses the generator and makes a forward pass (prediction) in noisy data.

        Args:
            z (np.array | tf.Tensor): Can either be a numpy array or a tensorflow tensor.

        Returns:
            A tensorflow array containing the generated data.

        """

        logger.info('Sampling with the model ...')

        # Performs the forward pass on the generator
        preds = self.G(z)

        return preds

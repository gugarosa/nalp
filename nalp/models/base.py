import tensorflow as tf

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class Model(tf.keras.Model):
    """A Model class is responsible for easily-implementing a neural network, when
    custom training or additional sets are not needed.

    """

    def __init__(self, name=''):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Model, self).__init__(name=name)

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def generate_text(self, start, length=100, temperature=1.0):
        """Generates text by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            start (str): The start string to generate the text.
            length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Returns:
            A list of generated text.

        """

        logger.debug(f'Generating text with length: {length} ...')

        # Encoding the start string into tokens
        start_tokens = self.encoder.encode(start)

        # Expanding the first dimension of tensor
        start_tokens = tf.expand_dims(start_tokens, 0)

        # Creating an empty list to hold the sampled_tokens
        sampled_tokens = []

        # Resetting the network states
        self.reset_states()

        # For every possible generation
        for i in range(length):
            # Predicts the current token
            preds = self(start_tokens)

            # Removes the first dimension of the tensor
            preds = tf.squeeze(preds, 0)

            # Regularize the prediction with the temperature
            preds /= temperature

            # Samples a predicted token
            sampled_token = tf.random.categorical(
                preds, num_samples=1)[-1, 0].numpy()

            # Put the sampled token back to the current token
            start_tokens = tf.expand_dims([sampled_token], 0)

            # Appends the sampled token to the list
            sampled_tokens.append(sampled_token)

        # Decodes the list into raw text
        text = self.encoder.decode(sampled_tokens)

        return text


class CustomModel(Model):
    """A CustomModel class is responsible for hardly-implementing a neural network, when
    custom training or additional settings are needed.

    """

    def __init__(self, name=''):
        """Initialization method.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(CustomModel, self).__init__(name=name)

    def compile(self, optimizer, loss):
        """Main building method.

        Args:
            optimizer (tf.optimizers): An optimizer instance.
            loss (tf.loss): A loss instance.

        """

        # Creates an optimizer object
        self.optimizer = optimizer

        # Defining the loss function
        self.loss = loss

        # Defining training accuracy metric
        self.train_accuracy = tf.metrics.CategoricalAccuracy(
            name='train_accuracy')

        # Defining training loss metric
        self.train_loss = tf.metrics.Mean(name='train_loss')

        # Defining validation accuracy metric
        self.val_accuracy = tf.metrics.CategoricalAccuracy(
            name='val_accuracy')

        # Defining validation loss metric
        self.val_loss = tf.metrics.Mean(name='val_loss')

    @tf.function
    def step(self, x, y):
        """Performs a single batch optimization step.

        Args:
            x (tf.Tensor): A tensor containing the inputs.
            y (tf.Tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self(x)

            # Calculate the loss
            loss = self.loss(y, preds)

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply gradients using an optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        # Update the loss metric state
        self.train_loss.update_state(loss)

        # Update the accuracy metric state
        self.train_accuracy.update_state(y, preds)

    @tf.function
    def val_step(self, x, y):
        """Performs a single batch evaluation step.

        Args:
            x (tf.Tensor): A tensor containing the inputs.
            y (tf.Tensor): A tensor containing the inputs' labels.

        """

        # Calculate the predictions based on inputs
        preds = self(x)

        # Calculate the loss
        loss = self.loss(y, preds)

        # Update the testing loss metric state
        self.val_loss.update_state(loss)

        # Update the testing accuracy metric state
        self.val_accuracy.update_state(y, preds)

    def fit(self, batches, val_batches=None, epochs=100):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing (x, y) pairs.
            val_batches (Dataset): Validation batches containing (x, y) pairs.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting states to further append losses and accuracies
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            # Iterate through all possible training batches, dependending on batch size
            for x_batch, y_batch in batches:
                # Performs the optimization step
                self.step(x_batch, y_batch)

            # Checks if there is a validation set
            if val_batches:
                # Iterate through all possible batches, dependending on batch size
                for x_batch, y_batch in val_batches:
                    # Evaluates the network
                    self.val_step(x_batch, y_batch)

            logger.info(f'Loss: {self.train_loss.result().numpy():.4f} | Accuracy: {self.train_accuracy.result().numpy():.4f} | Val Accuracy: {self.val_loss.result().numpy():.4f if val_batches else "?"} | Val Accuracy: {self.val_accuracy.result().numpy():.4f if val_batches else "?"}')

    def evaluate(self, val_batches):
        """Evaluates the model.

        Args:
            val_batches (Dataset): Validation batches containing (x, y) pairs.

        """

        logger.info('Evaluating model ...')

        # Resetting states to further append losses and accuracies
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

        # Iterate through all possible batches, dependending on batch size
        for x_batch, y_batch in val_batches:
            # Evaluates the network
            self.val_step(x_batch, y_batch)

        logger.info(
            f'Loss: {self.val_loss.result().numpy():.4f} | Accuracy: {self.val_accuracy.result().numpy():.4f}')

    @tf.function
    def predict(self, x):
        """Uses the model and makes a forward pass (prediction) in new data.

        Args:
            x (np.array | tf.Tensor): Can either be a numpy array or a tensorflow tensor.

        Returns:
            A tensorflow array containing the predictions. Note that if you use a softmax class in your model,
            these will be probabilities.

        """

        logger.info('Predicting with the model ...')

        # Performs the forward pass
        preds = self(x)

        return preds


class AdversarialModel(Model):
    """An AdversarialModel class is responsible for customly implementing
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
        super(AdversarialModel, self).__init__(name=name)

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
        z = tf.random.normal([x.shape[0], 1, 1, self.G.noise_dim])

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
                f'Loss(G): {self.G_loss.result().numpy()} | Loss(D): {self.D_loss.result().numpy()}')

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
        preds = self.G(z, training=False)

        return preds

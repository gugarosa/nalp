import tensorflow as tf
from tensorflow.keras import Model

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class ComplexNeural(Model):
    """A ComplexNeural class is responsible for hardly-implementing a neural network, when
    custom training or additional sets are needed.

    """

    def __init__(self, name):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., layers.

        Args:
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(ComplexNeural, self).__init__(name=name)

    def compile(self, optimize, loss, metrics=[]):
        """Main building method.

        Args:
            optimize (tf.optimizers): An optimizer instance.
            loss (tf.loss): A loss instance.
            metrics (list): A list of metrics to be displayed.

        """

        # Creates an optimizer object
        self.optimize = optimize

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
    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you will need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

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
        self.optimize.apply_gradients(
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
        for epoch in range(epochs):
            # Resetting states to further append losses and accuracies
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            # Iterate through all possible training batches, dependending on batch size
            for x_batch, y_batch in batches:
                # Performs the optimization step
                self.step(x_batch, y_batch)

            logger.debug(
                f'Epoch: {epoch+1}/{epochs} | Loss: {self.train_loss.result().numpy():.4f} | Accuracy: {self.train_accuracy.result().numpy():.4f}')

            # Checks if there is a validation set
            if val_batches:
                # Iterate through all possible batches, dependending on batch size
                for x_batch, y_batch in val_batches:
                    # Evaluates the network
                    self.val_step(x_batch, y_batch)

                logger.debug(
                    f'Val Loss: {self.val_loss.result().numpy():.4f} | Val Accuracy: {self.val_accuracy.result().numpy():.4f}\n')

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

    def generate_text(self, encoder, start, length=100, temperature=1.0):
        """Generates text by feeding to the network the
        current token (t) and predicting the next token (t+1).

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder.
            start (str): The start string to generate the text.
            length (int): Length of generated text.
            temperature (float): A temperature value to sample the token.

        Returns:
            A list of generated text.

        """

        logger.debug(f'Generating text with length: {length} ...')

        # Encoding the start string into tokens
        start_tokens = encoder.encode(start)

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
        text = encoder.decode(sampled_tokens)

        return text

import nalp.utils.logging as l
import numpy as np
import tensorflow as tf
from nalp.core.neural import Neural

logger = l.get_logger(__name__)


class RNN(Neural):
    """A RNN class is the one in charge of Recurrent Neural Networks vanilla implementation.
    
    References:
        http://psych.colorado.edu/~kimlab/Elman1990.pdf

    """

    def __init__(self, vocab_size=1, hidden_size=2, learning_rate=0.001):
        """Initialization method.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The amount of hidden neurons.
            learning_rate (float): A big or small addition on the optimizer steps.

        """

        logger.info('Overriding class: Neural -> RNN.')

        # Overrides its parent class with any custom arguments if needed
        super(RNN, self).__init__()

        # One for vocab size
        self._vocab_size = vocab_size

        # One for the amount of hidden neurons
        self._hidden_size = hidden_size

        # And the last for the learning rate
        self._learning_rate = learning_rate

        # Actually build the model
        self._build()

        logger.info('Class overrided.')

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @property
    def hidden_size(self):
        """int: The amount of hidden neurons.

        """

        return self._hidden_size

    @property
    def learning_rate(self):
        """float: A big or small addition on the optimizer steps.

        """

        return self._learning_rate

    def _build(self):
        """Main building method.

        """

        logger.info('Running private method: build().')

        # Builds the model layers
        self._build_layers()

        # Builds the learning objects
        self._build_learners()

        # Builds the metrics
        self._build_metrics()

        logger.info('Model ready to be used.')

    def _build_layers(self):
        """Builds the model layers itself.

        """

        logger.debug(
            f'Constructing model with shape: ({self.hidden_size}, {self.vocab_size}).')

        # Creates a simple RNN cell
        self.cell = tf.keras.layers.SimpleRNNCell(self.hidden_size)

        # Creates the RNN loop itself
        self.rnn = tf.keras.layers.RNN(self.cell)

        # Creates the linear (Dense) layer
        self.linear = tf.keras.layers.Dense(self.vocab_size)

        # And finally, a softmax activation for life's easing
        self.softmax = tf.keras.layers.Softmax()

    def _build_learners(self):
        """Builds all learning-related objects (i.e., loss and optimizer).

        """

        # Defining the loss function
        self.loss = tf.losses.CategoricalCrossentropy()

        logger.debug(f'Loss: {self.loss}.')

        # Creates an optimizer object
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        logger.debug(
            f'Optimizer: {self.optimizer} | Learning rate: {self.learning_rate}.')

    def _build_metrics(self):
        """Builds any desired metrics to be used with the model.

        """

        # Defining accuracy metric
        self.accuracy_metric = tf.metrics.CategoricalAccuracy(
            name='accuracy_metric')

        # Defining loss metric
        self.loss_metric = tf.keras.metrics.Mean(name='loss_metric')

        logger.debug(
            f'Accuracy: {self.loss_metric} | Mean Loss: {self.loss_metric}.')

    @tf.function
    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # We need to apply the input into the first recorrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        # Finally, we output its probabilites
        x = self.softmax(x)

        return x

    def _sample_from_multinomial(self, probs, temperature):
        """Samples an vocabulary index from a multinomial distribution.

        Args:
            probs (tf.Tensor): An tensor of probabilites.
            temperature (float): The amount of diversity to include when sampling.

        Returns:
            The index of sampled character or word.

        """

        # Converting to float64 to avoid multinomial distribution erros
        probs = np.asarray(probs).astype('float64')

        # Then, we calculate the log of probs, divide by temperature and apply
        # exponential
        exp_probs = np.exp(np.log(probs) / temperature)

        # Finally, we normalize it
        norm_probs = exp_probs / np.sum(exp_probs)

        # Sampling from multinomial distribution
        dist_probs = np.random.multinomial(1, norm_probs, 1)

        # The predicted index will be the argmax of the distribution
        pred_idx = np.argmax(dist_probs)

        return pred_idx

    def generate_text(self, dataset, start_text='', length=1, temperature=1.0):
        """Generates a maximum length of new text based on the probability of next char
        ocurring.

        Args:
            dataset (OneHot): A OneHot dataset object.
            start_text (str): The initial text for generating new text.
            length (int): Maximum amount of generated text.
            temperature (float): The amount of diversity to include when sampling.

        Returns:
            A list containing a custom generated text (can be characters or words).

        """

        logger.info(f'Generating new text with length: {length}.')

        # Defining variable to hold decoded generation
        output_text = start_text

        # Creating indexated tokens from starting text
        tokens_idx = dataset.indexate_tokens(
            list(start_text), dataset.vocab_index)

        # Creating seed to be inputed to the predictor
        seed = np.zeros(
            (1, len(tokens_idx), dataset.vocab_size), dtype=np.float32)

        # Iterate through maximum desired length
        for _ in range(length):
            # Iterate through all indexated tokens
            for i, idx in enumerate(tokens_idx):
                # Encodes each token into dataset's encoding
                seed[0, i] = dataset.encode(idx, dataset.vocab_size)

            # Calculates the prediction
            predict = self(seed).numpy()

            # Chooses a index based on the predictions probability distribution
            pred_idx = self._sample_from_multinomial(
                predict[-1], temperature)

            # Removing first indexated token
            tokens_idx = np.delete(tokens_idx, 0, 0)

            # Appending predicted index to the end of indexated tokens
            tokens_idx = np.append(tokens_idx, pred_idx)

            # Outputting generated characters to start text
            output_text.append(dataset.index_vocab[pred_idx])

        return output_text

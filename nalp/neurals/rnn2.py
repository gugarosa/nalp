import nalp.utils.logging as l
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

        # Creates the final linear (Dense) layer
        self.linear = tf.keras.layers.Dense(self.vocab_size)

    def _build_learners(self):
        """Builds all learning-related objects (i.e., loss and optimizer).

        """

        # Defining the loss function
        self.loss = tf.losses.CategoricalCrossentropy(from_logits=True)

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

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        """

        # We need to apply the input into the first recorrent layer
        x = self.rnn(x)

        # Finally, the input suffers a linear combination to output correct shape
        x = self.linear(x)

        return x

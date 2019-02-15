import nalp.utils.decorators as d
import nalp.utils.logging as l
import tensorflow as tf
import numpy as np
from nalp.core.neural import Neural

logger = l.get_logger(__name__)


class RNN(Neural):
    """A RNN class is the one in charge of Recurrent Neural Networks vanilla implementation.
    They were implemented using this paper: http://psych.colorado.edu/~kimlab/Elman1990.pdf

    Properties:
        max_length (int): The maximum length of the encoding.
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The amount of hidden neurons.
        learning_rate (float): A big or small addition on the optimizer steps.

    Methods:
        model(): tf.nn.dynamic_rnn with tf.nn.rnn_cell.BasicRNNCell
        loss(): tf.nn.softmax_cross_entropy_with_logits_v2
        accuracy(): tf.equal(tf.argmax)
        optimizer(): tf.train.AdamOptimizer
        predictor(): tf.argmax
        predictor_prob(): tf.nn.softmax
        train(input_batch, target_batch, epochs, verbose, save_model): Trains the network.
        predict(input_batch, model_path, probability): Predicts a new input.
        generate_text(dataset, start_text, length, model_path): Generates text beginning with a custom text seed.

    """

    def __init__(self, max_length=1, vocab_size=1, hidden_size=2, learning_rate=0.001, shape=None):
        """Initialization method.

        Args:
            max_length (int): The maximum length of the encoding.
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The amount of hidden neurons.
            learning_rate (float): A big or small addition on the optimizer steps.
            shape (list): A list containing in its first position the shape of the inputs (x)
            and on the second position, the shape of the labels (y).

        """

        logger.info('Overriding class: Neural -> RNN.')

        # Overrides its parent class with any custom arguments if needed
        super(RNN, self).__init__(shape=shape)

        # We need to create a property holding the max length of the encoding
        self._max_length = max_length
        # One for vocab size
        self._vocab_size = vocab_size
        # One for the amount of hidden neurons
        self._hidden_size = hidden_size
        # And the last for the learning rate
        self._learning_rate = learning_rate

        # The implemented methods should also be instanciated
        # Defines the model
        self.model
        # Calculates the loss function
        self.loss
        # Defines the accuracy function
        self.accuracy
        # Creates the optimization task
        self.optimizer
        # If you wish, predict new inputs based on indexes
        self.predictor
        # Or probabilities
        self.predictor_prob

        logger.info('Class overrided.')

    @property
    def max_length(self):
        """The maximum length of the encoding.

        """

        return self._max_length

    @property
    def vocab_size(self):
        """The size of the vocabulary.

        """

        return self._vocab_size

    @property
    def hidden_size(self):
        """The amount of hidden neurons.

        """

        return self._hidden_size

    @property
    def learning_rate(self):
        """A big or small addition on the optimizer steps.

        """

        return self._learning_rate

    @d.define_scope
    def model(self):
        """ The model should be constructed here. You can use whatever tensorflow
        operations you need.

        Returns:
            The model for further optimization and learning.

        """

        logger.debug(
            f'Constructing model with shape: ({self.hidden_size}, {self.vocab_size}).')

        # W will be the weight matrix
        self.W = tf.Variable(tf.random_normal(
            [self.hidden_size, self.vocab_size]))

        # b will the bias vector
        self.b = tf.Variable(tf.random_normal([self.vocab_size]))

        # For vanilla RNN, we will use a basic RNN cell
        self.cell = tf.keras.layers.SimpleRNNCell(self.hidden_size)

        # We do also need to create the RNN object itself
        self.o, self.h = tf.nn.dynamic_rnn(
            self.cell, self.x, dtype=tf.float32)

        # Transposing the vector dimensions
        self.o = tf.transpose(self.o, [1, 0, 2])

        # Gathering the re array
        self.o = self.o[-1]

        return tf.matmul(self.o, self.W) + self.b

    @d.define_scope
    def loss(self):
        """The loss function should be defined according your knowlodge.

        Returns:
            The loss function itself.

        """

        # Defining the loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.model, labels=self.y)

        # Applying an extra operation on defined loss
        loss = tf.reduce_mean(cross_entropy)

        logger.debug(f'Loss: {loss}.')

        return loss

    @d.define_scope
    def accuracy(self):
        """Calculates the accuracy between predicted and true labels

        Returns:
            The accuracy value itself.
        """

        # Defining the accuracy function
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1)), tf.float32))

        logger.debug(f'Accuracy: {accuracy}.')

        return accuracy

    @d.define_scope
    def optimizer(self):
        """An optimizer is the key of the learning task. Define your own
        as you may please.

        Returns:
            An optimizer object that minimizes the loss function.

        """

        # Creates an optimizer object
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        logger.debug(
            f'Optimizer: {optimizer} | Learning rate: {self.learning_rate}.')

        return optimizer.minimize(self.loss)

    @d.define_scope
    def predictor(self):
        """A predictor is responsible for returning an understable output.

        Returns:
            The index of the character with the highest probability of being the target.

        """

        # Creates a predictor object
        predictor = tf.cast(tf.argmax(self.model, 1), tf.int32)

        logger.debug(f'Predictor: {predictor}.')

        return predictor

    @d.define_scope
    def predictor_prob(self):
        """A predictor is responsible for returning an understable output.

        Returns:
            The probability array of an index being the target.
        """

        # Creates a probability predictor object
        predictor_prob = tf.nn.softmax(self.model)

        logger.debug(f'Predictor (prob): {predictor_prob}.')

        return predictor_prob

    def train(self, dataset, epochs=100, batch_size=1, verbose=0, save_model=1):
        """Trains a model.

        Args:
            dataset (Dataset): A Dataset object containing already encoded data (X, Y).
            epochs (int): The maximum number of training epochs.
            batch_size (int): The maximum size for each training batch.
            verbose (boolean): If verbose is true, additional printing will be done.
            save_model (boolean): If save_model is true, model will be saved into models/.

        """

        logger.info(f'Model ready to be trained for: {epochs} epochs.')
        logger.info(f'Batch size: {batch_size}.')

        # Initializing all tensorflow variables
        init = tf.global_variables_initializer()

        # Instanciating a new tensorflow session
        sess = tf.Session()

        # Running the first session's step
        sess.run(init)

        # Iterate through all epochs
        for epoch in range(epochs):
            # Creating lists to append losses and accuracies
            loss = []
            acc = []

            # Iterate through all possible batches, dependending on batch size
            for input_batch, target_batch in dataset.create_batches(dataset.X, dataset.Y, batch_size):
                # We run the session by feeding batches to it
                _, batch_loss, batch_acc = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict={
                    self.x: input_batch, self.y: target_batch})
                # Appending losses and accuracies
                loss.append(batch_loss)
                acc.append(batch_acc)

            # For each epoch, we need to calculate the mean of losses and accuracies (sum over batches)
            loss = np.mean(loss)
            acc = np.mean(acc)

            # If verbose is True, additional printing will be made
            if (verbose):
                logger.debug(
                    f'Epoch: {epoch}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.4f}')

        # If save model is True, we will save it for further restoring
        if (save_model):
            # Declaring a saver object for saving the model
            saver = tf.train.Saver()

            # Creating a custom string to be its output name
            self._model_path = f'models/rnn-hid{self.hidden_size}-lr{self.learning_rate}-e{epochs}-loss{loss:.4f}-acc{acc:.4f}'

            # Saving the model
            saver.save(sess, self._model_path)

            logger.info(f'Model saved: {self._model_path}.')

    def predict(self, input_batch, model_path=None, probability=1):
        """Predicts a new input based on a pre-trained network.

        Args:
            input_batch (tensor): An input batch to be predicted.
            model_path (str): A string holding the path to the desired model.
            probability (boolean): If true, will return a probability insteaf of a label.

        Returns:
            The index of the prediction.

        """

        # Declaring a saver object for saving the model
        saver = tf.train.Saver()

        # Instanciating a new tensorflow session
        sess = tf.Session()

        # Restoring the model, should use the same name as the one it was saved
        if (self._model_path):
            saver.restore(sess, self._model_path)
        else:
            self._model_path = model_path
            saver.restore(sess, model_path)

        logger.info(f'Model restored from: {self._model_path}.')
        logger.info(f'Predicting with probability={probability}.')

        # Running the predictor method according to argument
        if (probability):
            predict = sess.run([self.predictor_prob],
                               feed_dict={self.x: input_batch})
        else:
            predict = sess.run([self.predictor], feed_dict={
                               self.x: input_batch})

        return predict

    def generate_text(self, dataset, start_text='', length=1, model_path=None):
        """Generates a maximum length of new text based on the probability of next char
        ocurring.

        Args:
            dataset (OneHot): A OneHot object.
            start_text (str): The initial text for generating new text.
            length (int): Maximum amount of generated text.
            model_path (str): If needed, will load a different model from the previously trained.

        Returns:
            A list containing a custom generated text (can be characters or words).

        """

        logger.info(f'Generating new text with length: {length}.')

        # Declaring a saver object for saving the model
        saver = tf.train.Saver()

        # Instanciating a new tensorflow session
        sess = tf.Session()

        # Restoring the model, should use the same name as the one it was saved
        if (self._model_path):
            saver.restore(sess, self._model_path)
        else:
            saver.restore(sess, model_path)

        # Defining variable to hold decoded generation
        output_text = start_text

        # Creating indexated tokens from starting text
        tokens_idx = dataset.indexate_tokens(list(start_text), dataset.vocab_index)

        # Creating seed to be inputed to the predictor
        seed = np.zeros((1, len(tokens_idx), dataset.vocab_size), dtype=np.int32)

        # Iterate through maximum desired length
        for _ in range(length):
            # Iterate through all indexated tokens
            for i, idx in enumerate(tokens_idx):
                # Encodes each token into dataset's encoding
                seed[0, i] = dataset.encode(idx, dataset.vocab_size)

            # Calculates the prediction
            predict = sess.run([self.predictor_prob], feed_dict={self.x: seed})

            # Chooses a index based on the predictions probability distribution
            pred_idx = np.random.choice(
                range(dataset.vocab_size),
                p=predict[0][-1]
            )

            # Removing first indexated token
            tokens_idx = np.delete(tokens_idx, 0, 0)

            # Appending predicted index to the end of indexated tokens
            tokens_idx = np.append(tokens_idx, pred_idx)

            # Outputting generated characters to start text
            output_text.append(dataset.index_vocab[pred_idx])

        return output_text

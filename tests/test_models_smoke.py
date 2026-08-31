from importlib.util import find_spec

import pytest

tensorflow_available = find_spec("tensorflow") is not None
pytestmark = pytest.mark.skipif(
    not tensorflow_available, reason="TensorFlow is not installed"
)

if tensorflow_available:
    import tensorflow as tf

    from nalp.models import DCGAN, GAN, WGAN
    from nalp.models.discriminators import EmbeddedTextDiscriminator, TextDiscriminator
    from nalp.models.generators import GumbelLSTMGenerator, LSTMGenerator, RMCGenerator
    from nalp.models.layers import GumbelSoftmax, RelationalMemoryCell


def setup_function():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)


def test_image_models_forward_with_public_configuration():
    gan = GAN(input_shape=(4,), noise_dim=3, n_samplings=1, alpha=0.2)

    assert gan.G(tf.zeros((2, 3))).shape == (2, 4)
    assert gan.D(tf.zeros((2, 4))).shape == (2, 1)
    assert gan.G.noise_dim == 3
    assert gan.G.alpha == 0.2
    assert not hasattr(gan, "_G")
    assert not hasattr(gan, "_D")
    assert not hasattr(gan, "_history")

    dcgan = DCGAN(input_shape=(8, 8, 1), noise_dim=3, n_samplings=2)
    images = dcgan.G(tf.zeros((2, 1, 1, 3)), training=False)

    assert images.shape == (2, 8, 8, 1)
    assert dcgan.D(images, training=False).shape[0] == 2

    wgan = WGAN(
        input_shape=(8, 8, 1),
        noise_dim=3,
        n_samplings=2,
        model_type="gp",
        clip=0.02,
        penalty=4,
    )
    assert (wgan.model_type, wgan.clip, wgan.penalty_lambda) == ("gp", 0.02, 4)


def test_text_models_forward():
    tokens = tf.constant([[1, 2, 3], [2, 3, 4]])
    generator = LSTMGenerator(vocab_size=7, embedding_size=4, hidden_size=5)

    assert generator(tokens).shape == (2, 3, 7)
    generator.reset_state()

    gumbel = GumbelLSTMGenerator(vocab_size=7, embedding_size=4, hidden_size=5, tau=1.0)
    logits, probabilities, samples = gumbel(tokens)

    assert logits.shape == probabilities.shape == (2, 3, 7)
    assert samples.shape == (2, 3)

    embedded = EmbeddedTextDiscriminator(
        vocab_size=7,
        max_length=3,
        embedding_size=4,
        n_filters=(2,),
        filters_size=(2,),
        dropout_rate=0,
    )
    text = TextDiscriminator(
        max_length=3,
        embedding_size=4,
        n_filters=(2,),
        filters_size=(2,),
        dropout_rate=0,
    )

    assert embedded(tokens, training=False).shape == (2, 1, 2)
    assert text(tf.one_hot(tokens, 7), training=False).shape == (2, 1, 2)


def test_rmc_and_gumbel_config_round_trips():
    cell = RelationalMemoryCell(2, 2, 2, n_layers=1)
    outputs = tf.keras.layers.RNN(cell, return_sequences=True)(tf.ones((2, 3, 4)))
    clone = RelationalMemoryCell.from_config(cell.get_config())

    assert outputs.shape == (2, 3, 8)
    assert clone.state_size == [8, 8]
    assert clone.output_size == 8

    generator = RMCGenerator(
        vocab_size=7,
        embedding_size=4,
        n_slots=2,
        n_heads=2,
        head_size=2,
        n_layers=1,
    )
    assert generator(tf.constant([[1, 2, 3], [2, 3, 4]])).shape == (2, 3, 7)

    gumbel = GumbelSoftmax.from_config(GumbelSoftmax(axis=1).get_config())
    assert gumbel.axis == 1

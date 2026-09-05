import numpy as np
import pytest
import tensorflow as tf

from nalp.core import Generator
from nalp.encoders import IntegerEncoder
from nalp.models.generators import (
    BiLSTMGenerator,
    GRUGenerator,
    GumbelLSTMGenerator,
    GumbelRMCGenerator,
    LSTMGenerator,
    RMCGenerator,
    RNNGenerator,
    StackedRNNGenerator,
)


def setup_function():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)


def _generator(model_type):
    encoder = IntegerEncoder()
    encoder.learn({"a": 0, "b": 1, "c": 2}, {0: "a", 1: "b", 2: "c"})
    if issubclass(model_type, RMCGenerator):
        kwargs = {"n_slots": 2, "n_heads": 2, "head_size": 2, "n_layers": 1}
    else:
        kwargs = {"hidden_size": (4, 5) if model_type is StackedRNNGenerator else 4}
    return model_type(encoder=encoder, vocab_size=3, embedding_size=4, **kwargs)


def _logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


@pytest.mark.parametrize(
    "model_type",
    [
        RNNGenerator,
        GRUGenerator,
        LSTMGenerator,
        BiLSTMGenerator,
        StackedRNNGenerator,
        RMCGenerator,
        GumbelLSTMGenerator,
        GumbelRMCGenerator,
    ],
)
def test_greedy_generation_uses_logits_in_every_recurrent_family(
    model_type, monkeypatch
):
    model = _generator(model_type)
    model(tf.constant([[0]]))
    model.linear.kernel.assign(tf.zeros_like(model.linear.kernel))
    model.linear.bias.assign([0.0, 2.0, 1.0])
    monkeypatch.setattr(
        "nalp.models.layers.gumbel_softmax.gumbel_distribution",
        lambda shape: tf.broadcast_to([100.0, 0.0, 0.0], shape),
    )

    assert model.generate_greedy_search("a", max_length=2) == ["b", "b"]


@pytest.mark.parametrize("model_type", [GumbelLSTMGenerator, GumbelRMCGenerator])
def test_gumbel_temperature_sampling_uses_scaled_logits(model_type, monkeypatch):
    model = _generator(model_type)
    model(tf.constant([[0]]))
    model.linear.kernel.assign(tf.zeros_like(model.linear.kernel))
    model.linear.bias.assign([0.0, 2.0, 1.0])
    captured = []

    def sample(logits, count):
        captured.append(logits.numpy())
        return tf.zeros((1, count), tf.int64)

    monkeypatch.setattr(tf.random, "categorical", sample)

    assert model.generate_temperature_sampling("a", max_length=1, temperature=0.5) == [
        "a"
    ]
    assert model.tau == 0.5
    np.testing.assert_allclose(captured, [[[0.0, 4.0, 2.0]]])


class FixedGenerator(Generator):
    def __init__(self):
        super().__init__(name="fixed_generator")
        self.encoder = IntegerEncoder()
        self.encoder.learn({"a": 0, "b": 1, "c": 2}, {0: "a", 1: "b", 2: "c"})

    def call(self, inputs):
        return tf.math.log(tf.constant([[[0.3, 0.1, 0.6]]]))


@pytest.mark.parametrize(
    ("k", "p", "expected", "probabilities"),
    [
        (0, 0.7, "a", [2 / 3, 1 / 3]),
        (0, 0.5, "c", [1.0]),
        (0, 0.0, "b", [0.6, 0.3, 0.1]),
        (0, 1.0, "b", [0.6, 0.3, 0.1]),
        (2, 0.0, "a", [2 / 3, 1 / 3]),
        (2, 0.8, "a", [2 / 3, 1 / 3]),
    ],
)
def test_top_sampling_keeps_the_threshold_crossing_token_and_original_indices(
    k, p, expected, probabilities, monkeypatch
):
    captured = []

    def sample_last(logits, count):
        captured.append(tf.nn.softmax(logits).numpy())
        return tf.constant([[logits.shape[-1] - 1]], tf.int64)

    monkeypatch.setattr(tf.random, "categorical", sample_last)

    assert FixedGenerator().generate_top_sampling("a", max_length=1, k=k, p=p) == [
        expected
    ]
    np.testing.assert_allclose(captured[0], [probabilities], rtol=1e-6)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_temperature_sampling_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        FixedGenerator().generate_temperature_sampling("a", temperature=temperature)


@pytest.mark.parametrize(
    "kwargs", [{"k": -1}, {"p": -0.1}, {"p": 1.1}, {"p": float("nan")}]
)
def test_top_sampling_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        FixedGenerator().generate_top_sampling("a", **kwargs)


@pytest.mark.parametrize("model_type", [GumbelLSTMGenerator, GumbelRMCGenerator])
def test_temperature_assignment_updates_an_existing_graph_without_changing_weights(
    model_type, monkeypatch
):
    model = _generator(model_type)
    model.tau = 1.0
    inputs = tf.constant([[0]])
    model(inputs)
    model.linear.kernel.assign(tf.zeros_like(model.linear.kernel))
    model.linear.bias.assign([0.0, 2.0, 1.0])
    weights = model.get_weights()
    monkeypatch.setattr(
        "nalp.models.layers.gumbel_softmax.gumbel_distribution", tf.zeros
    )
    traced_call = tf.function(model.call)
    before = traced_call(inputs)[1]

    model.tau = 0.1
    after = traced_call(inputs)[1]

    assert model.tau == 0.1
    assert isinstance(model.tau, float)
    assert not np.allclose(before, after)
    np.testing.assert_allclose(after, tf.nn.softmax([[[0.0, 20.0, 10.0]]]))
    assert len(weights) == len(model.get_weights())
    for old, new in zip(weights, model.get_weights()):
        np.testing.assert_array_equal(old, new)
    with pytest.raises(ValueError, match="tau"):
        model.tau = 0
    assert model.tau == 0.1


@pytest.mark.parametrize("model_type", [RMCGenerator, GumbelRMCGenerator])
def test_rmc_preserves_chunked_state_and_restores_its_initializer(model_type):
    model = _generator(model_type)
    inputs = tf.constant([[0, 1, 2], [2, 1, 0]])
    first = _logits(model(inputs))
    model.reset_state()
    after_reset = _logits(model(inputs))
    model.reset_states()
    chunked = tf.concat([_logits(model(inputs[:, i : i + 1])) for i in range(3)], 1)

    np.testing.assert_allclose(after_reset, first, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(chunked, first, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("model_type", "parent_type"),
    [(GumbelLSTMGenerator, LSTMGenerator), (GumbelRMCGenerator, RMCGenerator)],
)
def test_gumbel_temperature_does_not_change_the_keras_weight_schema(
    model_type, parent_type, tmp_path
):
    reference = _generator(parent_type)
    model = _generator(model_type)
    inputs = tf.constant([[0, 1, 2]])
    reference(inputs)
    model(inputs)
    path = tmp_path / "generator.weights.h5"
    reference.save_weights(path)

    model.load_weights(path)
    reference.reset_state()
    model.reset_state()
    np.testing.assert_allclose(_logits(model(inputs)), reference(inputs), rtol=1e-6)

    model.tau = 0.5
    model.save_weights(path)
    restored = _generator(model_type)
    restored(inputs)
    restored.load_weights(path)
    assert restored.tau == 5.0
    restored.reset_state()
    reference.reset_state()
    np.testing.assert_allclose(_logits(restored(inputs)), reference(inputs), rtol=1e-6)

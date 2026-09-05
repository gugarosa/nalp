import numpy as np
import pytest
import tensorflow as tf

from nalp.models import DCGAN, GAN, GSGAN, WGAN, MaliGAN, RelGAN, SeqGAN


def setup_function():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)


def _discrete_model(model_type):
    model = model_type(
        vocab_size=3,
        max_length=3,
        embedding_size=4,
        hidden_size=5,
        n_filters=(2,),
        filters_size=(1,),
        dropout_rate=0,
    )
    model.compile(
        tf.keras.optimizers.SGD(0.01),
        tf.keras.optimizers.SGD(0.01),
        tf.keras.optimizers.SGD(0.01),
    )
    return model


def _discriminator(logits):
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Lambda(
                lambda inputs: tf.gather(logits, inputs[:, 0])[:, None, :]
            )
        ]
    )


def test_maligan_rewards_are_normalized_real_class_odds():
    model = _discrete_model(MaliGAN)
    logits = np.log([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
    inputs = tf.constant([[0, 0], [1, 1], [2, 2]])
    expected = np.repeat((np.array([4.0, 0.25, 1.0]) / 5.25)[:, None], 2, axis=1)

    for shift in (0.0, 10.0):
        model.D = _discriminator(logits + shift)
        np.testing.assert_allclose(model._get_reward(inputs), expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        ([[0.0, 0.0], [0.0, 0.0]], [0.5, 0.5]),
        ([[1000.0, -1000.0], [-1000.0, 1000.0]], [1.0, 0.0]),
    ],
)
def test_maligan_rewards_remain_finite_for_equal_and_extreme_logits(logits, expected):
    model = _discrete_model(MaliGAN)
    model.D = _discriminator(logits)
    rewards = model._get_reward(tf.constant([[0, 0], [1, 1]]))

    np.testing.assert_allclose(rewards, np.repeat(np.array(expected)[:, None], 2, 1))
    np.testing.assert_allclose(tf.reduce_sum(rewards, axis=0), [1.0, 1.0])


@pytest.mark.parametrize("n_rollouts", [1, 2, 3])
def test_seqgan_rewards_preserve_sample_and_timestep_axes(n_rollouts):
    model = _discrete_model(SeqGAN)
    model.D = _discriminator(np.log([[0.2, 0.8], [0.8, 0.2]]))
    inputs = tf.constant([[0, 0, 0], [1, 1, 1]])

    rewards = model._get_reward(inputs, n_rollouts)

    np.testing.assert_allclose(rewards, [[0.2] * 3, [0.8] * 3])


def test_seqgan_rollouts_use_only_the_prefix_and_original_start(monkeypatch):
    model = _discrete_model(SeqGAN)
    model.T = 0.5
    inputs = tf.constant([[0, 1, 2], [1, 2, 0]])
    start = tf.constant([[2], [0]])
    model.G(inputs)
    model.G.linear.kernel.assign(tf.zeros_like(model.G.linear.kernel))
    model.G.linear.bias.assign([0.0, 1.0, 2.0])
    calls = []
    sampling_logits = []
    original_call = model.G.call

    def record_call(tokens):
        calls.append(tokens.numpy().copy())
        return original_call(tokens)

    def sample(logits, count, dtype):
        sampling_logits.append(logits.numpy())
        return tf.zeros((2, count), dtype=dtype)

    monkeypatch.setattr(model.G, "call", record_call)
    monkeypatch.setattr(tf.random, "categorical", sample)
    model._get_reward(inputs, n_rollouts=1, start_tokens=start)

    np.testing.assert_array_equal(calls[0], tf.concat([start, inputs[:, :1]], 1))
    np.testing.assert_allclose(sampling_logits, [[[0.0, 2.0, 4.0]] * 2] * 3)


def test_seqgan_rejects_zero_rollouts():
    model = _discrete_model(SeqGAN)
    with pytest.raises(ValueError, match="n_rollouts"):
        model._get_reward(tf.constant([[0, 1, 2]]), n_rollouts=0)


def test_seqgan_pretraining_records_generator_loss():
    model = _discrete_model(SeqGAN)
    x = tf.constant([[0, 1, 2], [1, 2, 0]])
    y = tf.constant([[1, 2, 0], [2, 0, 1]])

    model.pre_fit(tf.data.Dataset.from_tensors((x, y)), g_epochs=1, d_epochs=0)

    assert model.G_loss.result().numpy() > 0
    assert model.history["pre_G_loss"] == [model.G_loss.result().numpy()]


@pytest.mark.parametrize("model_type", [MaliGAN, SeqGAN])
@pytest.mark.parametrize("stage", ["pre_fit", "fit"])
def test_discrete_training_scores_targets_not_input_context(
    model_type, stage, monkeypatch
):
    model = _discrete_model(model_type)
    real_x = tf.zeros((2, 3), tf.int32)
    real_y = tf.ones((2, 3), tf.int32)
    fake_x = tf.zeros((2, 3), tf.int32)
    fake_y = tf.fill((2, 3), 2)
    discriminator_batches = []
    reward_inputs = []

    def discriminator_step(x, y):
        discriminator_batches.append((x.numpy(), y.numpy()))
        model.D_loss.update_state(1.0)

    def reward(x, *args, **kwargs):
        reward_inputs.append(x.numpy())
        if model_type is SeqGAN:
            np.testing.assert_array_equal(kwargs["start_tokens"], fake_x[:, :1])
        return tf.ones_like(x, dtype=tf.float32)

    def generator_step(x, y, rewards):
        np.testing.assert_array_equal(x, fake_x)
        np.testing.assert_array_equal(y, fake_y)
        model.G_loss.update_state(1.0)

    monkeypatch.setattr(model, "generate_batch", lambda *args: (fake_x, fake_y))
    monkeypatch.setattr(model, "D_step", discriminator_step)
    monkeypatch.setattr(model, "_get_reward", reward)
    monkeypatch.setattr(model, "G_step", generator_step)
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array([0, 2]))
    batches = tf.data.Dataset.from_tensors((real_x, real_y))

    if stage == "pre_fit":
        model.pre_fit(batches, g_epochs=0, d_epochs=1)
    else:
        model.fit(batches, epochs=1, d_epochs=1)
        np.testing.assert_array_equal(reward_inputs, [fake_y.numpy()])

    assert discriminator_batches
    for tokens, labels in discriminator_batches:
        np.testing.assert_array_equal(tokens, [[1, 1, 1], [2, 2, 2]])
        np.testing.assert_array_equal(labels, [0, 1])


@pytest.mark.parametrize("model_type", [MaliGAN, SeqGAN])
def test_generator_update_restarts_the_sampled_context(model_type):
    model = _discrete_model(model_type)
    x = tf.constant([[0, 1, 2], [1, 2, 0]])
    y = tf.constant([[1, 2, 0], [2, 0, 1]])
    model.G(x)
    model.G.reset_state()
    expected = tf.reduce_mean(model.loss(y, model.G(x)))
    for state in model.G.rnn.states:
        state.assign(tf.ones_like(state) * 5)

    model.G_step(x, y, tf.ones_like(x, dtype=tf.float32))

    np.testing.assert_allclose(model.G_loss.result(), expected, rtol=1e-6)


@pytest.mark.parametrize("model_type", [MaliGAN, SeqGAN])
def test_discrete_models_pretrain_and_fit_one_batch(model_type):
    model = _discrete_model(model_type)
    x = tf.constant([[0, 1, 2], [1, 2, 0]])
    y = tf.constant([[1, 2, 0], [2, 0, 1]])
    batches = tf.data.Dataset.from_tensors((x, y))
    model.pre_fit(batches, g_epochs=1, d_epochs=1)
    kwargs = {"n_rollouts": 2} if model_type is SeqGAN else {}
    model.fit(batches, epochs=1, d_epochs=1, **kwargs)

    for losses in model.history.values():
        assert len(losses) == 1
        assert np.isfinite(losses).all()
    assert all(np.isfinite(weight.numpy()).all() for weight in model.weights)


@pytest.mark.parametrize("model_type", [GSGAN, RelGAN])
def test_gumbel_gans_train_after_pretraining(model_type):
    kwargs = (
        {"hidden_size": 4}
        if model_type is GSGAN
        else {
            "max_length": 3,
            "n_slots": 2,
            "n_heads": 2,
            "head_size": 2,
            "n_layers": 1,
            "n_filters": (2,),
            "filters_size": (1,),
            "dropout_rate": 0,
        }
    )
    model = model_type(vocab_size=3, embedding_size=4, **kwargs)
    model.compile(
        tf.keras.optimizers.SGD(0.01),
        tf.keras.optimizers.SGD(0.01),
        tf.keras.optimizers.SGD(0.01),
    )
    x = tf.constant([[0, 1, 2], [1, 2, 0]])
    y = tf.constant([[1, 2, 0], [2, 0, 1]])
    model.G_pre_step(x, y)
    before = [weight.numpy().copy() for weight in model.G.trainable_variables]
    model.step(x, y)

    assert np.isfinite(model.G_loss.result())
    assert np.isfinite(model.D_loss.result())
    assert all(np.isfinite(weight.numpy()).all() for weight in model.weights)
    assert any(
        not np.array_equal(old, new.numpy())
        for old, new in zip(before, model.G.trainable_variables)
    )


@pytest.mark.parametrize("kind", ["gan", "dcgan", "wgan-wc", "wgan-gp"])
def test_image_models_train_one_batch(kind):
    if kind == "gan":
        model = GAN(input_shape=(4,), noise_dim=3, n_samplings=1)
        inputs = tf.random.normal((2, 4))
    else:
        kwargs = {"model_type": kind[-2:]} if kind.startswith("wgan") else {}
        model_type = WGAN if kwargs else DCGAN
        model = model_type(
            input_shape=(8, 8, 1), noise_dim=3, n_samplings=2, dropout_rate=0, **kwargs
        )
        inputs = tf.random.normal((2, 8, 8, 1))
    model.compile(tf.keras.optimizers.SGD(0.001), tf.keras.optimizers.SGD(0.001))

    if isinstance(model, WGAN):
        model.D_step(inputs)
        model.G_step(inputs)
    else:
        model.step(inputs)

    assert np.isfinite(model.G_loss.result())
    assert np.isfinite(model.D_loss.result())
    assert all(np.isfinite(weight.numpy()).all() for weight in model.weights)

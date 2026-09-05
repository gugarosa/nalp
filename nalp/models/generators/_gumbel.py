"""Shared temperature state and inference for Gumbel generators."""

import math

import tensorflow as tf


class GumbelGeneratorMixin:
    """Keep Gumbel relaxation separate from logits-based token sampling."""

    @property
    def tau(self) -> float:
        """Gumbel-Softmax temperature."""

        return self._tau

    @tau.setter
    def tau(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("tau must be finite and positive.")

        if hasattr(self, "_tau_tensor"):
            self._tau_tensor.assign(value)
        else:
            # A resource makes assignments visible to traced training steps.
            # Keep it out of Keras weights to preserve the existing weight schema.
            self._tau_tensor = tf.Variable(
                value, dtype=tf.float32, trainable=False, name="tau"
            )
        self._tau = value

    def _generation_logits(self, x: tf.Tensor) -> tf.Tensor:
        return self(x)[0]

    def generate_temperature_sampling(
        self, start: str, max_length: int = 100, temperature: float = 1.0
    ) -> list[str]:
        """Sample from logits while retaining the public temperature assignment."""

        self.tau = temperature
        return super().generate_temperature_sampling(start, max_length, temperature)

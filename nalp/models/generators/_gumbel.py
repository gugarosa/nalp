# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

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
        """Set the relaxation temperature for eager and traced generator calls.

        Args:
            value: Finite positive temperature mirrored by a nontrainable float32 resource.

        Raises:
            ValueError: The temperature is non-finite or not positive.

        """

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"`tau` must be finite and positive, but got {value}.")

        if hasattr(self, "_tau_tensor"):
            self._tau_tensor.assign(value)
        else:
            # Resource assignments reach traced steps without changing the existing Keras weight schema
            self._tau_tensor = tf.Variable(value, dtype=tf.float32, trainable=False, name="tau")
        self._tau = value

    def _generation_logits(self, x: tf.Tensor) -> tf.Tensor:
        return self(x)[0]

    def generate_temperature_sampling(
        self, start: str | list[str], max_length: int = 100, temperature: float = 1.0
    ) -> list[str]:
        """Sample from logits while retaining the public temperature assignment.

        Assign tau before sampling, reset recurrent state, and stop at an end-of-sentence token or the length limit.
        Token sampling uses raw logits rather than the Gumbel-Softmax probabilities.

        Args:
            start: Character string or pre-tokenized prompt matching the encoder.
            max_length: Maximum number of generated tokens without counting the prompt.
            temperature: Finite positive divisor for logits and assigned Gumbel relaxation temperature.

        Returns:
            Decoded sampled tokens, including an end-of-sentence marker when one is generated.

        Raises:
            ValueError: The temperature is non-finite or not positive.
            RuntimeError: The attached encoder has not learned a mapping.

        """

        self.tau = temperature
        return super().generate_temperature_sampling(start, max_length, temperature)

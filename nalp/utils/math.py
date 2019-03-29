import numpy as np


def sample_from_multinomial(probs, temperature):
    """Samples an index from a multinomial distribution.

    Args:
        probs (tf.Tensor): A tensor of probabilites.
        temperature (float): The amount of diversity to include when sampling.

    Returns:
        An index of the sampled object.

    """

    # Converting to float64 to avoid multinomial distribution erros
    probs = np.asarray(probs).astype('float64')

    # Then, we calculate the log of probs, divide by temperature and apply exponential
    exp_probs = np.exp(np.log(probs) / temperature)

    # Finally, we normalize it
    norm_probs = exp_probs / np.sum(exp_probs)

    # Sampling from multinomial distribution
    dist_probs = np.random.multinomial(1, norm_probs, 1)

    # The predicted index will be the argmax of the distribution
    pred_idx = np.argmax(dist_probs)

    return pred_idx

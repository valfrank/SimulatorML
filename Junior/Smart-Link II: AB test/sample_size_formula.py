import numpy as np
from scipy import stats


def calculate_sample_size(
        reward_avg: float, reward_std: float, mde: float, alpha: float, beta: float
) -> int:
    """Calculate sample size.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    mde: float :
        minimum detectable effect
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    int :
        sample size

    """
    assert mde > 0, "mde should be greater than 0"

    # Implement your solution here
    sample_size = int((2 * np.square(stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(1-beta)) * np.square(reward_std)) / (mde * reward_avg))

    return sample_size


def calculate_mde(
        reward_std: float, sample_size: int, alpha: float, beta: float
) -> float:
    """Calculate minimal detectable effect.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    sample_size: int :
        sample size
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    float :
        minimal detectable effect

    """

    # Implement your solution here
    mde = ((stats.norm.ppf(1- alpha/2) + stats.norm.ppf(1-beta)) * np.sqrt(2) * reward_std) / np.sqrt(sample_size)

    return mde
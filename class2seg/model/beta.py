import numpy as np
from scipy.stats import beta


class BetaMixtureModel:
    def __init__(
        self,
        alpha1: float = 0.5,
        beta1: float = 2.0,
        alpha2: float = 2.0,
        beta2: float = 0.5,
    ):
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    def fit(self, data: np.ndarray, print_estimates: bool = False):

        for it in range(
            0, 100
        ):  # TODO: Might not converage in 100 runs, check likelihood
            d1 = beta.pdf(data, self.alpha1, self.beta1)
            d2 = beta.pdf(data, self.alpha2, self.beta2)

            m2 = self.alpha2 / (self.alpha2 + self.beta2)
            th_4d2 = beta.ppf(0.9, self.alpha1, self.beta1)

            d1[data > m2] = 0
            d2[data < th_4d2] = 0

            w1 = d1.sum()
            w2 = d2.sum()

            d1 *= w1
            d2 *= w2

            d = d1 + d2
            d[d == 0] = 1e-6  # avoid division by zero

            d1 /= d
            d2 /= d

            mu1 = sum(d1 * data) / sum(d1) + 1e-6
            std1 = np.sqrt(sum(d1 * np.square(data - mu1)) / sum(d1)) + 1e-6

            mu2 = sum(d2 * data) / sum(d2) + 1e-6
            std2 = np.sqrt(sum(d2 * np.square(data - mu2)) / sum(d2)) + 1e-6

            self.alpha1 = ((1 - mu1) / std1**2 - 1 / mu1) * mu1**2
            self.beta1 = self.alpha1 * (1 / mu1 - 1)

            self.alpha2 = ((1 - mu2) / std2**2 - 1 / mu2) * mu2**2
            self.beta2 = self.alpha2 * (1 / mu2 - 1)

        if print_estimates:
            print(
                "alpha1: {:2.6}, beta1: {:2.6}, alpha2: {:2.6}, beta2: {:2.6}".format(
                    self.alpha1, self.beta1, self.alpha2, self.beta2
                )
            )

    def predict(self, data: np.ndarray) -> np.ndarray:
        d = self.predict_proba(data)
        return np.argmax(d, axis=1)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:

        d1 = beta.pdf(data, self.alpha1, self.beta1)
        d2 = beta.pdf(data, self.alpha2, self.beta2)

        # if d2 has its peak not at 1, d1 might have larger values, so let's add d1 here
        # (both, d1 and d2 have extreme small values here)
        beyond_d2_peak = data > beta.mean(self.alpha2, self.beta2) + np.sqrt(
            beta.var(self.alpha2, self.beta2)
        )
        d2[beyond_d2_peak] += d1[beyond_d2_peak]
        d_sum = d1 + d2
        d1 /= d_sum
        d2 /= d_sum
        d = np.concatenate((d1[:, np.newaxis], d2[:, np.newaxis]), axis=1)
        return d

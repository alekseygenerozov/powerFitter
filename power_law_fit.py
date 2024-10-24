import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore


def power_law_pdf(
    pow_index: float, observed_data, min_value: float, max_value: float
) -> float:
    """
    Power law pdf between min_value and max_value, with power-law index pow_index

    :param pow_index:  Power law index * (-1)
    :param observed_data:  Observed data (Array-like)
    :param min_value: Inner truncation of power law
    :param max_value: Outer truncation of power law

    :return: PDF evaluated evaluated at observed data
    """
    return (
        (1 - pow_index)
        * (observed_data) ** (-pow_index)
        / (max_value ** (1 - pow_index) - min_value ** (1 - pow_index))
    )


def sample_power_law(pow_index: float, min_value: float, max_value: float) -> float:
    """
    Generate a random number from a power law PDF with power law index -pow_index.
    between min_value and max_value. It does this by passing a random number between 0 and 1
    to the corresponding inverse CDF.

    :param pow_index:  power law index * (-1)
    :param min_value: Inner truncation of power law
    :param max_value: Outer truncation of power law

    :return: Random number from power law distribution.
    """
    r = np.random.uniform()

    if pow_index == 1:
        return min_value * (max_value / min_value) ** r
    else:
        return (
            r * (max_value ** (1.0 - pow_index) - min_value ** (1.0 - pow_index))
            + min_value ** (1.0 - pow_index)
        ) ** (1.0 / (1 - pow_index))


def negative_log_likelihood(
    pow_index: float, observed_data, min_value: float, max_value: float
) -> float:
    """
    Negative log of likelihood

    :param pow_index:  power law index * (-1)
    :param observed_data:  Observed data (Array-like)

    :return: Negative log likelihood corresponding to pow_index
    """
    log_likely = -np.sum(
        np.log(power_law_pdf(pow_index, observed_data, min_value, max_value))
    )

    return log_likely


def validate_and_clean(
    observed_data, min_value: Optional[float] = None, max_value: Optional[float] = None
):
    """
    Validate and clean observed_data. Make sure observed_data consists of posive real numbers. Perform sanity checks on
    min_value and max_value. Also filter data to be in the range [min_value, max_value]

    :param observed_data: Observed data
    :param min_value: Inner truncation of power law (If None use minimum of data), defaults to None
    :param max_value: Outer truncation of power law (If None use maximum of data), defaults to None

    :return: Tuple with filtered and cleaned observed_data, min_value, and max_value.
    """
    if not all(isinstance(x, (int, float)) for x in observed_data):
        raise ValueError("Observed data must be positive, real numbers.")
    observed_data = np.array(observed_data).astype(float)
    if np.any(np.isnan(observed_data) | (observed_data <= 0.0)):
        raise ValueError("Observed data must be positive, real numbers.")
    ##If input value is not specified then use the minimum / maximum of the data
    if min_value is None:
        min_value = np.min(observed_data)
    if max_value is None:
        max_value = np.max(observed_data)
    if (min_value < 0) or (min_value >= max_value):
        raise ValueError(
            "min_value must be less than or equal to max_value, and both must be postive."
        )

    observed_data = observed_data[
        (observed_data <= max_value) & (observed_data >= min_value)
    ]
    if len(observed_data) <= 1:
        raise ValueError("Too few points in domain for fitting.")

    return observed_data, min_value, max_value


def fit_power(
    observed_data,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """
    Maximum likelihood power law fit to observed_data (must be positive
    real numbers).

    The inner and outer truncation can either be specified via min_value and max_value keyword arguments.
    These are none by default, which means the truncations are determined from the data.
    The fitting is always performed on the subset of data on the interval [min_value, max_value].

    :param observed_data: Observed data
    :param min_value: Inner truncation of power law (If None use minimum of data), defaults to None
    :param max_value: Outer truncation of power law (If None use maximum of data), defaults to None

    :return: Best fit power-law index
    """
    ##Input validation
    observed_data, min_value, max_value = validate_and_clean(
        observed_data, min_value=min_value, max_value=max_value
    )
    ##Find power law between pow_min and pow_max that minimizes the negative log-likelihood.
    soln = minimize_scalar(
        negative_log_likelihood,
        bracket=(-4.0, 4.0),
        args=(observed_data, min_value, max_value),
    )
    gbest = soln["x"]

    return gbest


def fit_power_with_error(
    observed_data, min_value: Optional[float] = None, max_value: Optional[float] = None
) -> Tuple[float, float]:
    """
    Maximum likelihood power law fit (w. uncertainty) to observed_data (must be positive
    real numbers). Uncertainty is estimated via the bootstrap method.

    The inner and outer truncation can either be specified via min_value and max_value keyword arguments.
    These are none by default, which means the truncations are determined from the data.
    The fitting is always performed on the subset of data on the interval [min_value, max_value].

    :param observed_data: Observed data
    :param min_value: Inner truncation of power law (If None use minimum of data), defaults to None
    :param max_value: Outer truncation of power law (If None use maximum of data), defaults to None

    :return: Best fit power-law index with error estimate
    """
    fits_all = []
    for ii in range(100):
        re_samp = np.random.choice(observed_data, size=len(observed_data))
        gbest = fit_power(re_samp, min_value=min_value, max_value=max_value)
        fits_all.append(gbest)

    return np.mean(fits_all), np.std(fits_all)


def main():
    parser = argparse.ArgumentParser(description="Fit a power-law to data.")
    parser.add_argument("data_file", help="Data location")
    parser.add_argument(
        "--min_value", type=float, default=None, help="Inner truncation for power law"
    )
    parser.add_argument(
        "--max_value", type=float, default=None, help="Outer truncation for power law"
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Flag to turn on uncertainty estimate.",
    )

    args = parser.parse_args()
    ##Read data from file
    data = np.atleast_1d(np.genfromtxt(args.data_file)).ravel()
    ##Raise error if we have single datapoint or none
    if len(data) <= 1:
        raise ValueError("Too few points in domain for fitting.")
    ##If uncertainty flag is set, give best fit and uncertainty
    if args.uncertainty:
        res = fit_power_with_error(
            data, min_value=args.min_value, max_value=args.max_value
        )
        print(f"Best fit power law:{res[0]}, Uncertainty:{res[1]}")
    ##Else just give best fit
    else:
        res = fit_power(data, min_value=args.min_value, max_value=args.max_value)
        print(f"Best fit power law:{res}")


if __name__ == "__main__":
    main()

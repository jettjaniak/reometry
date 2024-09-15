#! /usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom


def prob_same_region(n, r, k):
    mean = k / n
    p = r / (r + mean)
    prob = 0
    size = 1
    while True:
        new_prob = prob
        for _ in range(int(2 * mean)):
            size += 1
            pr_size = nbinom.pmf(size, r, p, loc=1)
            pr_cond = pr_size * (size - 1) / (k - 1)
            new_prob += pr_cond
        if new_prob == prob:
            break
        prob = new_prob
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", default=2, type=int)
    parser.add_argument("-k", default=25_000, type=int, help="Number of samples")
    args = parser.parse_args()

    for n in range(50, 300, 10):
        print(f"n = {n}, prob = {prob_same_region(n, args.r, args.k)}")


if __name__ == "__main__":
    main()

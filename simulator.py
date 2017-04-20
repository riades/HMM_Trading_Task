import numpy as np
from scipy.stats import norm, expon


class SeriesSampler(object):
    def __init__(self, tan_mean, tan_var, intensity, wiener_var):
        self.__tan_mean = tan_mean
        self.__tan_var = tan_var
        self.__wiener_var = wiener_var
        self.__lambda = intensity

    def __sample_wiener(self, size):
        sample = norm.rvs(scale=self.__wiener_var, size=size)
        return np.cumsum(sample)

    def __sample_trend(self, size, bias=0):

        cursor = 0.0
        cum_sum = bias
        trend = []

        while cursor < size:
            previous = cursor
            cursor += expon.rvs(scale=self.__lambda)
            cursor = min(cursor, size)

            tan = norm.rvs(loc=self.__tan_mean,
                           scale=self.__tan_var)

            left_idx = np.int(np.floor(previous))
            right_idx = np.int(np.floor(cursor))

            supp = np.arange(left_idx, right_idx) - previous

            if len(supp) > 0:
                trend.append(tan * supp + cum_sum)

            cum_sum += tan * (cursor - previous)

        trend = np.hstack(trend)
        return trend

    def simulate(self, series_size, trend_bias=0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        noise = self.__sample_wiener(series_size)
        trend = self.__sample_trend(series_size, trend_bias)
        return {"series": np.maximum(trend + noise, 0), "trend": trend, "noise": noise}

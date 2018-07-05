from math import log
from collections import deque
from numpy import std


class EMA:

    def __init__(self, period):

        self.period = period
        self.ema_val = None
        self.alpha = 1 / (period + 1)
        self.aux_alpha = 1 - self.alpha

    def ema(self, value):

        if self.ema_val is None:

            self.ema_val = value

        return (value * self.alpha) + (self.ema_val * self.aux_alpha)


class Bbands:

    def __init__(self):

        self.period = 21
        self.aux = 1 / 21
        self.sum = 0
        self.last_el = deque()

    def sma(self, value):

        self.last_el.append(value)

        if len(self.last_el) >= self.period:

            res = self.sum * self.aux
            offset = 2 * std(self.last_el)
            self.sum -= self.last_el.popleft()
            return (res - offset, res + offset)

        return None


class OBV:

    def __init__(self):

        self.last_obv = 0
        self.last_price = None

    def get_obv(self, price, volume):

        if self.last_price is None:

            self.last_price = price

        if price > self.last_price:

            self.last_obv += volume

        elif price < self.last_price:

            self.last_obv -= volume

        self.last_price = price
        return self.last_obv


class Momentum:

    def __init__(self):

        self.momentum = 0

    def get(self, new_value):

        if new_value < 0 and self.momentum > -3:

            self.momentum -= 1

        elif new_value > 0 and self.momentum < 3:

            self.momentum += 1

        return self.momentum


def log_return(open_t, close_t, open_tm1, close_tm1):

    if open_t == 0.0:
        ret_t = 1.0
    else:
        ret_t = (close_t - open_t) / open_t

    if open_tm1 == 0.0:
        ret_tm1 = 1.0
    else:
        ret_tm1 = (close_tm1 - open_tm1) / open_tm1

    if ret_tm1 == 0.0:

        ret_tm1 += 0.0001

    aux = ret_t / ret_tm1

    if aux < 0.0:

        aux *= (-1)

    elif aux == 0.0:

        aux += 0.0001

    return log(aux)

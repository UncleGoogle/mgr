from time import time
from functools import wraps


def timeit(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        t_1 = int(time() * 1000)
        output = function(*args, **kwargs)
        t_2 = int(time() * 1000)
        print(f'<timeit> {function.__name__} runs: ', t_2 - t_1, 'ms')
        return output
    return wrapper

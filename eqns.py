"""Functions to support curve_fitting project."""

import numpy

def r_squared(x_data, y_data, function, *popt) -> float:  # asterick allows to unpack a tuple and pass as argument
    """Function for calculating R^2 of a particular model and data."""

    # first, calculate the residual sum of squares
    l = []
    for x, y in zip(x_data, y_data):
        diff = y - function(x, *popt)
        l.append(diff ** 2)
    residual_square_sum = sum(l)

    # next, calculate the total sum of squares
    y_mean = numpy.mean(y_data)
    l = []
    for num in y_data:
        l.append((num - y_mean) ** 2)
    total_square_sum = sum(l)

    # r_squared is 1 - residual sum of squares divided by total sum of squares
    return (total_square_sum - residual_square_sum) / total_square_sum
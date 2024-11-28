import random
import logging

import xgboost as xgb
import pandas as pd
import numpy as np

def get_range_data(num_row, rank, num_workers):
    """
    compute the data range based on the input data size and worker id
    :param num_row: total number of dataset
    :param rank: the worker id
    :param num_workers: total number of workers
    :return: begin and end range of input matrix
    """
    num_per_partition = int(num_row/num_workers)

    x_start = rank * num_per_partition
    x_end = (rank + 1) * num_per_partition

    if x_end > num_row:
        x_end = num_row

    return x_start, x_end

def read_train_data(rank: int, num_workers: int) -> xgb.DMatrix:
    """
    This function should load the data for worker @rank given that there is @num_workers workers

    For this test project this function generate random data but feel free to put your data laoding here
    """
    
    random.seed(71)
    # Generate data
    logging.info(f"Generating some data for rank={rank}")
    num_classes = 3
    class_distributions = [
        (
            (random.random()*100, 1),
            (x%3, random.random()),
            (x%2, random.random()+1),
            (x%2 + random.random(), random.random()),
            (x/3, random.random()),
            (x%2, random.random()),
        )
        for x in range(num_classes)
    ]

    X = []
    Y = []

    for _ in range(1000000):
        y = random.randint(0, num_classes-1)
        x = [
            random.random(),
            random.random() * 10,
            *[random.normalvariate(mu, sigma) for (mu, sigma) in class_distributions[y]]
        ]

        Y.append(y)
        X.append(x)
    logging.info("End of data generation")
    X = np.array(X)
    Y = np.array(Y)

    start, end = get_range_data(len(x), rank, num_workers)
    x = X[start:end, :]
    y = Y[start:end]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    return xgb.DMatrix(data=x, label=y)
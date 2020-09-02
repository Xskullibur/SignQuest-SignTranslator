import pandas as pd
import numpy as np

JSON_FILE = './input/WLASL_v0.3.json'
OUTPUT_FILE = './output/WLASL.csv'


def read_JSON():
    pd.read_json(JSON_FILE)


if __name__ == '__main__':
    read_JSON()
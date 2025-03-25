import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

from load import *

def main():
    data = load_file("PFC_con_4.csv")
    print(data.head())

if __name__ == "__main__":
    main()
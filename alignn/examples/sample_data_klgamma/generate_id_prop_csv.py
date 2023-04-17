#!/bin/env python3

import numpy as np
from glob import glob

ID_PROP_FILE = 'id_prop.csv'

def main():
    
    data = []
    files = glob('*.vasp')
    
    for file in files:
        data.append((file,
                     str(20*np.random.exponential()), 
                     str(np.random.exponential())) )

    with open(ID_PROP_FILE, 'w') as f:
        for xy in data:
            f.write(','.join(xy) + '\n')


if __name__ == '__main__':
    main()

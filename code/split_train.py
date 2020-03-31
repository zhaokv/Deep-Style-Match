import time
import random as rd

from tqdm import *

def main():
    trainSetPath = '../data/train_set.txt'
    trainSetAfterPath = '../data/train_set_after.txt'
    trainSetValPath = '../data/train_set_val.txt'
    valRatio = 0.2
    trainSetData = open(trainSetPath).readlines()
    trainSetAfterFile = open(trainSetAfterPath, 'w')
    trainSetValFile = open(trainSetValPath, 'w')
    for i in range(0, len(trainSetData)/2):
        if rd.random()<=valRatio:
            trainSetValFile.write(trainSetData[i*2])
            trainSetValFile.write(trainSetData[i*2+1])
        else:
            trainSetAfterFile.write(trainSetData[i*2])
            trainSetAfterFile.write(trainSetData[i*2+1])


if __name__=='__main__':
    main()

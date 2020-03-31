def main():
    matchsetFilePath = '../data/dim_fashion_matchsets.txt'
    matchPairPath = '../data/match_pair.txt'

    with open(matchsetFilePath, 'r') as matchsetFile:
        matchsetData = matchsetFile.readlines()
    matchSet = set()
    for line in matchsetData:
        tmp = line.split()[1].split(';')
        for i in range(0, len(tmp)):
            for j in range(i+1, len(tmp)):
                tmpA = tmp[i].split(',')
                tmpB = tmp[j].split(',')
                for itemA in tmpA:
                    for itemB in tmpB:
                        matchSet.add((itemA, itemB))
                        matchSet.add((itemB, itemA))

    with open(matchPairPath, 'w') as matchPairFile:
        for (itemA, itemB) in matchSet:
            matchPairFile.write(itemA+','+itemB+'\n')
   
if __name__=='__main__':
    main()

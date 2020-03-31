from tqdm import *

def main():
    jsonPath='../data/amazon/meta_Clothing_Shoes_and_Jewelry.json'
    jsonData = open(jsonPath).readlines()
    itemDic = {}
    matchPairSet = set()
    spliter=','
    for line in tqdm(jsonData):
        try:
            metadata=eval(line)
            if 'related' not in metadata or 'also_bought' not in metadata['related']:
                continue
            asin=metadata['asin']
            title=metadata['title']
            coPurchase=metadata['related']['also_bought']
            itemDic[asin]=spliter.join([word.lower() for word in title.split()])
            for item in coPurchase:
                matchPairSet.add(asin+','+item)
                matchPairSet.add(item+','+asin)
        except:
            continue
    print('items: %s' % len(itemDic), 'pairs: %s' % len(matchPairSet))
    #upperBound=int(len(itemDic)*0.2)
    upperBound=int(len(itemDic)*1)
    outItemSet=set(itemDic.keys()[0:upperBound])
    print(list(outItemSet)[0:10])
    print(list(matchPairSet)[0:10])
    #itemsPath='../data/amazon/dim_items.txt'
    #pairPath='../data/amazon/match_pair.txt'
    itemsPath='../data/amazon/dim_items_all.txt'
    pairPath='../data/amazon/match_pair_all.txt'
    with open(itemsPath, 'w') as f:
        for item in itemDic:
            if item not in outItemSet:
                continue
            f.write(item+' 1 '+itemDic[item]+'\n')
    with open(pairPath, 'w') as f:
        for pair in matchPairSet:
            tmp=pair.split(',')
            if tmp[0] not in outItemSet or tmp[1] not in outItemSet:
                continue
            f.write(pair+'\n')

if __name__=='__main__':
    main()

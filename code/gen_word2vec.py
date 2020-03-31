import gensim, logging

import config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO)

def genWord2Vec(itemPath, matchItemPath, modelPath):
    matchSet = set()
    for line in open(matchItemPath).readlines():
        itemId = line.split()[0]
        matchSet.add(itemId)
    sentences = []
    for line in open(itemPath).readlines():
        tmp = line.split()
        if tmp[0] in matchSet:
            sentences.append(tmp[2].split(','))
    model = gensim.models.Word2Vec(sentences, 
            size=config.w2vSize, 
            min_count = 0, workers=4)
    model.save(modelPath)

def main():
    dataDir = '../data/'
    itemPath = dataDic+'dim_items.txt'
    matchItemPath = dataDic+'match_item.txt'
    modelPath = dataDic+'word2vec.model'
    genWord2Vec(itemPath, matchItemPath, modelPath)

if __name__=='__main__':
    main()

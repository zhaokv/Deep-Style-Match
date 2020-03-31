import sys
caffeRoot = '/home/zhaokui/code/caffe/'
sys.path.insert(0, caffeRoot+'python')
import caffe

import numpy as np

blobPath = './item_mean.binaryproto'
arrayPath = './item_mean.npy'
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(blobPath, 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save(arrayPath, out)

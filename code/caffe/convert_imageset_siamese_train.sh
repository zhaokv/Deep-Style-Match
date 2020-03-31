rm -rf train_leveldb
/home/zhaokui/code/caffe/build/tools/convert_imageset_siamese \
        /home/zhaokui/research/KDD/data/taobao/ \
        /home/zhaokui/research/KDD/data/taobao/pro_train_set.txt \
        train_leveldb --backend="leveldb" --resize_width=256 --resize_height=256

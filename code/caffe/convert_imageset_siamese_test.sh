rm -rf test_leveldb
/home/zhaokui/code/caffe/build/tools/convert_imageset_siamese \
        /home/zhaokui/research/KDD/data/taobao/ \
        /home/zhaokui/research/KDD/data/taobao/pro_test_set.txt \
        test_leveldb --backend="leveldb" --resize_width=256 --resize_height=256

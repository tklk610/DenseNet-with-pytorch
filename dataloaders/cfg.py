# -*- coding:utf-8 -*-


##数据集的类别
NUM_CLASSES = 2

#数据集的存放位置
DATASET_DIR       = r'/media/jzdyjy/EEB2EF73B2EF3EA9/workstore/lekang/DenseNet/densenet_v0/data'
TRAIN_DATASET_DIR = r'/media/jzdyjy/EEB2EF73B2EF3EA9/workstore/lekang/DenseNet/densenet_v0/data/train'
VAL_DATASET_DIR   = r'/media/jzdyjy/EEB2EF73B2EF3EA9/workstore/lekang/DenseNet/densenet_v0/data/val'
TEST_DATASET_DIR  = r'/media/jzdyjy/EEB2EF73B2EF3EA9/workstore/lekang/DenseNet/densenet_v0/inference/image'

# DATASET_DIR        = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data'
# TRAIN_DATASET_DIR  = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/train'
# VAL_DATASET_DIR    = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/data/val'
# TEST_DATASET_DIR   = r'F:/lekang/PycharmPreject/NeralNetwork/DenseNet/densenet_lk/inference/image'


#这里需要加入自己的最终预测对应字典，例如:'0': '花'
labels_to_classes = {
    '0' : 'OK',
    '1' : 'NG'
}

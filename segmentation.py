import os
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from detectron2.projects import point_rend

from detectron2.projects.point_rend import add_pointrend_config

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from  matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # 从config file 覆盖配置
    #     cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("coco_train_coco6",)  # 训练数据集名称
    cfg.DATASETS.TEST = ("coco_val_coco6",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程

    #    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 640  # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = 640  # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 768)  # 训练图片输入的最小尺寸，可以设定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = 640
    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING，其存在两种配置，分别为 choice 与 range ：
    # range 让图像的短边从 512-768随机选择
    # choice ： 把输入图像转化为指定的，有限的几种图片大小进行训练，即短边只能为 512或者768
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'

    cfg.MODEL.RETINANET.NUM_CLASSES = 5  # 类别数+1（因为有background）
    # cfg.MODEL.WEIGHTS="/home/yourstorePath/.pth"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 预训练模型权重
    cfg.SOLVER.IMS_PER_BATCH = 4  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size

    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    # 9000为你的训练数据的总数目，可自定义
    ITERS_IN_ONE_EPOCH = int(9000 / cfg.SOLVER.IMS_PER_BATCH)

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1  # 12 epochs，
    # 初始学习率
    cfg.SOLVER.BASE_LR = 0.002
    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9
    # 权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"
    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    # cfg.TEST.EVAL_PERIOD = 100

    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg.DATASETS.TRAIN)
    default_setup(cfg, args)
    return cfg


WINDOW_NAME = "COCO detections"


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file("/usr/local/work_surface/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.DATASETS.TRAIN = ("coco_train_coco8",)
    cfg.MODEL.DEVICE = 'cpu'
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.WEIGHTS = "model_final.pth"  # 你下载的模型地址和名称
    cfg.freeze()
    print(cfg.DATASETS.TRAIN)
    print(cfg.MODEL.RETINANET.NUM_CLASSES)
    return cfg

def toclassname(id):
    if id==0: return '梁柱钢筋'
    if id==1:return '木模板'
    if id==3: return '平铺钢筋'
    if id==4: return '混凝土'
    return '其他类别'

def add(tensor,list1):
    print(tensor.shape)
    object1=torch.nonzero(tensor,out=None,as_tuple = False)
    object1=object1.tolist()
    for i in range(len(object1)):
        if object1[i] not in list1:
            list1.append(object1[i])
    print(1111)



    # print(object1[0])
    # print(object1[1])
    # print(object1.shape)
    # per=object1.shape[0]/(tensor.shape[0]*tensor.shape[1])
    # print('{:.2%}'.format(per))
    # return '{:.2%}'.format(per)

def segmente(path):
    cfg = setup_cfg()
    print(cfg.DATASETS.TRAIN)
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(path, 1)
    im2 = im[:, :, ::-1]  # transform image to rgb
    plt.imshow(im2)
    cv2.imwrite('test.jpg',im2)
    plt.show()
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    classes = outputs['instances'].pred_classes.tolist()
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5,
                   instance_mode=ColorMode.SEGMENTATION)
    print(outputs['instances'].pred_masks.shape)
    mask = outputs['instances'].pred_masks
    size1 = mask[0].shape[0] * mask[0].shape[1]

    dict1_category = {}

    tensor_wood_p = None

    tensor_f_steel = None
    tensor_concrete = None

    for p in range(len(classes)):


        label = classes[p]
        print(label)

        if label == 1:
            if tensor_wood_p == None:
                tensor_wood_p = torch.nonzero(mask[p], out=None, as_tuple=False)
            # print('classes is ' + toclassname(classes[p]))
            else:
                torch.cat((tensor_wood_p, torch.nonzero(mask[p], out=None, as_tuple=False)), 0)
        if label == 3:
            if tensor_f_steel == None:
                tensor_f_steel = torch.nonzero(mask[p], out=None, as_tuple=False)
            # print('classes is ' + toclassname(classes[p]))
            else:
                torch.cat((tensor_f_steel, torch.nonzero(mask[p], out=None, as_tuple=False)), 0)
        if label == 4:
            if tensor_concrete == None:
                tensor_concrete = torch.nonzero(mask[p], out=None, as_tuple=False)
            # print('classes is ' + toclassname(classes[p]))
            else:
                torch.cat((tensor_concrete, torch.nonzero(mask[p], out=None, as_tuple=False)), 0)

    if tensor_wood_p != None:
        tensor_wood_p = torch.unique(tensor_wood_p, dim=0)

        dict1_category['wood_p'] = tensor_wood_p.shape[0]/size1
    else:
        dict1_category['wood_p'] = 0

    if tensor_f_steel != None:
        tensor_f_steel = torch.unique(tensor_f_steel, dim=0)
        print(tensor_f_steel[0])
        print(tensor_f_steel[1])

        dict1_category['f_steel'] = tensor_f_steel.shape[0]/size1
    else:
        dict1_category['f_steel'] = 0

    if tensor_concrete != None:
        tensor_concrete = torch.unique(tensor_concrete, dim=0)

        dict1_category['f_concrete'] = tensor_concrete.shape[0]/size1
    else:
        dict1_category['f_concrete'] = 0
    print(dict1_category)
    

    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # plt.imshow(out.get_image()[:, :, ::-1])
    # cv2.imwrite("1.png", out.get_image()[:, :, ::-1])
    # plt.show()

    return (dict1_category)





path1='/home/surf/jlb/train_img'
path3='/home/surf/jlb/val_img'
path2='/home/surf/jlb/surf/0715/'
train_root =path1 #数据集路径
val_root= path3

ann_root1='/home/surf/jlb/train.json'
ann_root2='/home/surf/jlb/val.json'#标注文件

CLASS_NAMES=['h_steel','wood','11','falt_steel','concrete','wodden','steeel11']
COLORS=[(255,0,0),(0,255,0),(100,100,100),(0,0,255),(0,0,0),(22,22,22),(33,33,33)]


def plain_register_dataset():
    # 训练集
    DatasetCatalog.register("coco_train_coco8", lambda: load_coco_json(ann_root2, val_root))
    MetadataCatalog.get("coco_train_coco8").set(
        thing_classes=CLASS_NAMES,
        thing_colors=COLORS,
        evaluator_type='coco')

    # DatasetCatalog.register("coco_val_coco8", lambda: load_coco_json(ann_root2, val_root))
    # MetadataCatalog.get("coco_val_coco8").set(
    #     evaluator_type='coco',
    #     json_file=ann_root1,
    #     image_root=train_root)


from detectron2.utils.visualizer import ColorMode
import cv2 as cv

if __name__ == '__main__':
    plain_register_dataset()
    cfg = setup_cfg()
    print(cfg.DATASETS.TRAIN)
    segmente("img_1.png")
    # predictor = DefaultPredictor(cfg)
    # path_d = '/home/surf/jlb/surf/0715/'
    # file11 = os.listdir(path_d)
    # for i in range(1):
    #     path = os.path.join(path_d, file11[i])
    #     im = cv2.imread(path, 1)
    #     im2 = im[:, :, ::-1]  # transform image to rgb
    #     plt.imshow(im2)
    #     plt.show()
    #     outputs = predictor(im)
    #     classes = outputs['instances'].pred_classes.tolist()
    #     # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5,
    #     #                instance_mode=ColorMode.SEGMENTATION)
    #     # print(outputs['instances'].pred_masks.shape)
    #     mask = outputs['instances'].pred_masks
    #     size1 = mask[0].shape[0] * mask[0].shape[1]



    #     dict1_category={}


    #     tensor_wood_p=None

    #     tensor_f_steel = None
    #     tensor_concrete = None



    #     for p in range(len(classes)):


    #         label=classes[p]

    #         if label == 1:
    #             if tensor_wood_p ==None:
    #                 tensor_wood_p=torch.nonzero(mask[p], out=None, as_tuple=False)
    #             # print('classes is ' + toclassname(classes[p]))
    #             else:torch.cat((tensor_wood_p,torch.nonzero(mask[p], out=None, as_tuple=False)),0)
    #         if label == 3:
    #             if tensor_f_steel ==None:
    #                 tensor_f_steel=torch.nonzero(mask[p], out=None, as_tuple=False)
    #             # print('classes is ' + toclassname(classes[p]))
    #             else:torch.cat((tensor_f_steel, torch.nonzero(mask[p], out=None, as_tuple=False)), 0)
    #         if label == 4:
    #             if tensor_concrete ==None:
    #                 tensor_concrete=torch.nonzero(mask[p], out=None, as_tuple=False)
    #             # print('classes is ' + toclassname(classes[p]))
    #             else: torch.cat((tensor_concrete, torch.nonzero(mask[p], out=None, as_tuple=False)), 0)

    #     if tensor_wood_p !=None:
    #         tensor_wood_p = torch.unique(tensor_wood_p,dim=0)
    #         dict1_category['wood_p']=tensor_wood_p.shape[0]/size1

    #     else:dict1_category['wood_p']=0

    #     if tensor_f_steel !=None:
    #         tensor_f_steel =torch.unique(tensor_f_steel,dim=0)
    #         print(tensor_f_steel[0])
    #         print(tensor_f_steel[1])
    #         dict1_category['f_steel'] = tensor_f_steel.shape[0]/size1
    #     else:dict1_category['f_steel']=0

    #     if tensor_concrete!=None:
    #         tensor_concrete = torch.unique(tensor_concrete,dim=0)

    #         dict1_category['f_concrete'] = tensor_concrete.shape[0]/size1
    #     else: dict1_category['f_concrete']=0
    #     print(dict1_category)








    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     plt.imshow(out.get_image()[:, :, ::-1])
    #     cv2.imwrite("1.png", out.get_image()[:, :, ::-1])
    #     plt.show()
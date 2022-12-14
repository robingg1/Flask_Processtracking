from datetime import datetime
import numpy as np
import torch
import os
from dateutil import rrule
import time

from segmentation import segmente
from classifyResNet import resnet_eval
from houghline import Hoff

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



def evaluate(path):
    rough_c = 0
    detailed_s = 0
    # 大致分类
    category = resnet_eval(True, path)

    result = segmente(path)
    print(result)
    
    if result['f_concrete'] > 0.29:
        rough_c = 70
        p_res=result['f_concrete'] * 1.4 
        res = result['f_concrete'] * 1.4 * 25 + rough_c
        return 1,p_res,res*0.01
    
    # 第一阶段 脚手架+钢筋
    if category == 0:
        rough_c = 0
        detailed_s = Hoff(path)
        print(detailed_s*20)
        p_res=detailed_s
        res=20 * detailed_s+rough_c
        # 霍夫处理
    if category == 1:
        rough_c = 70
        p_res=result['f_concrete'] * 1.4 
        res = result['f_concrete'] * 1.2 * 25 + rough_c
        return 3 ,p_res,res*0.01

    if category == 2:
        rough_c = 50
        p_res=result['f_steel'] * 1.5 
        res = result['f_steel'] * 1.3 * 25 + rough_c
        
    if category ==3:
        rough_c = 20
        p_res=result['wood_p'] * 1.4 
        res = result['wood_p'] * 1.4 * 25 + rough_c
        return 1, p_res,res*0.01
    print(f'category is {category}, percentage is {res}')
    res=res*0.01
    return category,p_res,res

       




def main(path):
    print(evaluate(path))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # path_d = '/home/surf/jlb/surf/0715/'
    # file11 = os.listdir(path_d)
    # for i in range(221, 223):
    #     path = os.path.join(path_d, file11[i])
    print(main('/usr/local/work_surface/flaskProject/pics/31011500991320016917/20221010093004.jpg'))




  

    # print(evaluate(path))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

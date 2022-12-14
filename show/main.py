from ctypes import resize
from datetime import datetime, timedelta
from os.path import join
import numpy as np
import cv2
import calendar
from PIL import Image, ImageDraw, ImageFont
import os
text = []


def puttext_list(img, text_position, font_size, font_file='yuyang_w4.ttf', align='TL', BG=None):
    print('start to put list')
    pilimg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilimg)

    for tp in text_position:
        text, position, color = tp[0], tp[1], tp[2]  # 文字，位置，颜色，尺寸，字体，对齐
        if len(tp) >= 4:
            font_size_ = tp[3]
        else:
            font_size_ = font_size

        if len(tp) >= 5:
            font_file_ = tp[4]
            if font_file_ is None or font_file_ == '':
                font_file_ = font_file
        else:
            font_file_ = font_file

        if len(tp) >= 6:
            align_ = tp[5]
        else:
            align_ = align
     

        font = ImageFont.truetype(os.path.join('show/font', font_file_), font_size_, encoding="utf-8")
       
        tw, th = text_size = draw.textsize(text, font)
        x, y = position
        x_start, y_start = x, y
        if 'T' in align_:
            y_start = y
        if 'B' in align_:
            y_start = y - th
        if 'L' in align_:
            x_start = x
        if 'R' in align_:
            x_start = x - tw
        if align_[0] == 'C':
            y_start = y - th // 2
        if align_[1] == 'C':
            x_start = x - tw // 2
        if BG is not None:
            draw.rectangle((x_start, y_start, x_start + tw, y_start + th), fill=BG, outline=BG)
        draw.text((x_start, y_start), str(text), fill=color, font=font)

    im_t = np.array(pilimg)
    return im_t.astype(np.uint8)


def draw(year,month,path_photo,floors,day_list,stage,save_path,percentage,latest_floors,pre_time,show_rate):
    d = 400
    img = np.ones((1079, 1920, 3), dtype="uint8") * 255
    # 生成白色背景
    # img = cv2.line(img, (0, 0), (1054, 1573), (57, 25, 14), 2)
    img = cv2.rectangle(img, (0, 0), (1920, 1079), (57, 25, 14), -1)
    # cv2.circle(img, (50, 50), 25, (255, 255, 0), -1)
    #
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, 'OpenCV', (0, 200), font, 3, (0, 0, 255), 15)
    # cv2.putText(img, 'OpenCV', (0, 200), font, 3, (0, 255, 0), 5)
    # 设置左上顶点和右下顶点，颜色，线条宽度
    # cv2.rectangle(img, (82, 124), (1556, 1028), (57, 25, 14), -1)
    cv2.rectangle(img, (82, 150), (1556, 1050), (69, 35, 11), 2)
    cv2.rectangle(img, (106, 654), (808, 694), (69, 35, 11), -1)
    cv2.rectangle(img, (832, 212), (1536, 722), (69, 35, 11), -1)
    cv2.rectangle(img, (106, 774), (810, 1036), (69, 35, 11), -1)
    cv2.rectangle(img, (832, 774), (1536, 1036), (69, 35, 11), -1)
    cv2.rectangle(img, (1576, 150), (1896, 1050), (69, 35, 11), 2)

    # 左下进度条

    text1 = []

    def draw_process_rate(days, floors, time):
        print('start draw process rate')
        list1 = days
        if len(list1) > 5:
            list1 = days[-5:]
            floors = floors[-5:]

        print(list1)
        r_length = 595
        x1 = 125
        y1 = 840
        cv2.line(img, (112, y1 - 15), (798, y1 - 15), (255, 255, 255), 1)
        text1.append(["每层施工统计", (136, 740), (255, 255, 255), 20])
        text1.append(["阶段完成图像", (848, 740), (255, 255, 255), 20])
        text1.append(["楼层", (136, 790), (255, 255, 255), 20])
        text1.append(["施工天数", (230, 790), (255, 255, 255), 20])
        for i in range(min(len(list1),len(floors))):
            cv2.rectangle(img, (x1, y1), (x1 + 65, y1 + 27), (105, 51, 18), -1)
            text1.append([f"{floors[-i-1]} 层", (x1 + 20, y1 + 3), (255, 255, 255), 18])
            # cv2.rectangle(img, (x1, y1), (x1 + 65, y1 + 25), (0,0,0), 1)
            length = int(r_length * (list1[-i-1] / 12))
            cv2.rectangle(img, (x1 + 65, y1), (x1 + 65 + length, y1 + 27), (160, 78, 27), -1)
            text1.append([f"{list1[-i-1]} 天", (x1 + 70, y1 + 3), (255, 255, 255), 18])
            time1 = time[-i-1]
            if len(time)>i:
                time1 = time[-i-1]
                text1.append([time1, (300 - 38, y1 + 2), (255, 255, 255), 16])
            y1 += 38

        img2 = puttext_list(img, text1, 17)
        return img2

    # list1 = [10, 6, 7, 7, 3]
    list1=day_list
    img = draw_process_rate(list1, floors, time=pre_time)

    text2 = []

    def draw_stage(list2, time,img_list):  # 绘画左下角天数进度
        # params: 前五层天数, 完成日期
        # 前半段进度
        
        cv2.rectangle(img, (844, 796), (1006, 826), (80, 104, 110), -1)
        cv2.rectangle(img, (1016, 796), (1180, 826), (73, 74, 135), -1)
        cv2.rectangle(img, (1188, 796), (1350, 826), (119, 52, 89), -1)
        cv2.rectangle(img, (1360, 796), (1520, 826), (184, 139, 48), -1)

        # 阶段文字
        text2.append(["上三层", (858, 800), (255, 255, 255), 17])
        text2.append(["上两层", (1075, 800), (255, 255, 255), 17])
        text2.append(["上一层", (1225, 800), (255, 255, 255), 17])
        text2.append(["此层", (1403, 800), (255, 255, 255), 17])

        # 前半段框
        cv2.rectangle(img, (844, 839), (1006, 946), (255, 255, 255), 1)
        cv2.rectangle(img, (1016, 839), (1180, 946), (255, 255, 255), 1)
        cv2.rectangle(img, (1188, 839), (1350, 946), (255, 255, 255), 1)
        cv2.rectangle(img, (1360, 839), (1520, 946), (255, 255, 255), 1)

        # 后半段进度
        
        cv2.rectangle(img, (844, 983), (844 + 114, 1013), (80, 104, 110), -1)
        cv2.rectangle(img, (1016, 983), (1016 + 114, 1013), (73, 74, 135), -1)
        cv2.rectangle(img, (1188, 983), (1188 + 114, 1013), (119, 52, 89), -1)
        cv2.rectangle(img, (1360, 983), (1360 + 114, 1013), (184, 139, 48), -1)
        
        # size(114,67)
        list_pos=[844,1016,1188,1360]
        list_pos1=[1006,1180,1350,1520]
        print(img_list)
        for i in range(4):
            print(len(img_list))
            if len(img_list)>i:
                img1_f=cv2.imread(img_list[-1-1*i])
                # cv2.resize(img_main_pic, (670, 434))
                img1_f=cv2.resize(img1_f, (list_pos1[-i-1]-list_pos[-i-1],107 ))
                img[839:946,list_pos[-i-1]:list_pos1[-i-1], ]=img1_f
                
            
        



        # 完成度
        text2.append([f'完成度: {list2[0]}%', (846, 984), (255, 255, 255), 16])
        text2.append([f'完成度: {list2[1]}%', (1019, 984), (255, 255, 255), 16])
        text2.append([f'完成度: {list2[2]}%', (1191, 984), (255, 255, 255), 16])
        text2.append([f'完成度: {list2[3]}%', (1363, 984), (255, 255, 255), 16])

        # 左上角底部文字说明
        text2.append(['施工阶段：', (125, 694), (255, 255, 255), 16])

        cv2.rectangle(img, (214, 694), (214 + 20, 694 + 20), (80, 104, 110), -1)
        cv2.rectangle(img, (405, 694), (405 + 20, 694 + 20), (73, 74, 135), -1)
        cv2.rectangle(img, (507, 694), (507 + 20, 694 + 20), (119, 52, 89), -1)
        cv2.rectangle(img, (658, 694), (658 + 20, 694 + 20), (184, 139, 48), -1)

        text2.append(["脚手架+柱钢筋", (244, 695), (255, 255, 255), 17])
        text2.append(["铺模板", (435, 695), (255, 255, 255), 17])
        text2.append(["铺平面钢筋", (537, 695), (255, 255, 255), 17])
        text2.append(["浇筑混凝土", (688, 695), (255, 255, 255), 17])

        text2.append(["抓拍图像", (879, 181), (255, 255, 255), 20])

        time = pre_time
        pos = [846, 1019, 1191, 1363]

        for i in range(4):
            if len(time)>i:
                time1 = time[-i-1]
                str1=time1
                text2.append([str1, (pos[-i-1], 953), (255, 255, 255), 16])

    def draw_main_down(stage, rate):
        print(stage,rate)
        rate=round(rate,2)
        # 现在进度和完成度
        # params: 进度index, 现在阶段完成度
        colors = [(80, 104, 110), (73, 74, 135),(119, 52, 89), (184, 139, 48),]
        stages = ['脚手架+钢筋',  '铺模板','平铺钢筋','浇筑混凝土']
        cv2.rectangle(img, (1238, 684), (1238 + 20, 684 + 20), colors[stage], -1)
        text2.append([f'{stages[stage]}', (1265, 684), colors[stage], 19])
        text2.append([f'完成度：{rate}%', (1390, 684), colors[stage], 19])
        img2 = puttext_list(img, text2, 13)
        return img2

    # 抓拍图像
    cv2.rectangle(img, (850, 230), (1522, 666), (255, 255, 255), 1)

    # 日历
    cv2.rectangle(img, (106, 212), (812, 722), (69, 35, 11), -1)
    cv2.line(img, (106, 252), (812, 252), (91, 52, 19), 1)
    cv2.line(img, (106, 680), (812, 680), (91, 52, 19), 1)
    # 竖线
    x0_final = x0 = x00 = 106
    x1_final = x1 = x11 = 812
    # 横线
    y0_final = y0 = y00 = 252
    y1_final = y1 = y11 = 680

    for i in range(6):
        x0 += int((x1_final - x0_final) / 7)
        cv2.line(img, (x0, y0_final), (x0, y1_final), (91, 52, 19), 1)
    for i in range(5):
        y0 += int((y1_final - y0_final) / 6)
        cv2.line(img, (x0_final, y0), (x1_final, y0), (91, 52, 19), 1)

    for i in range(7):
        weekday = ["一", "二", "三", "四", "五", "六", "日"]
        text.append([f"星期{weekday[i]}", (x00 + 22, 212 + 10), (124, 118, 110), 18])
        x00 += (x1_final - x0_final) / 7

    def draw_day(row, column, number, color_type):
        # 文字，位置，颜色，尺寸，字体，对齐
        x = int(x0_final + (column - 1) * (x1_final - x0_final) / 7) + 12
        y = int(y0_final + (row - 1) * (y1_final - y0_final) / 6) + 10
        color = (202, 192, 187) if color_type == 1 else (92, 58, 38)
        size = 32
        text.append([str(number), (x, y), color, size])

    def draw_calendar(year, month):
        text.append([f"{year}年{month}月", (150, 176), (236, 233, 231), 18])
        weekday, total_day = calendar.monthrange(year, month)
        date = datetime.date(datetime(year=year, month=month, day=1))
        weekday = date.isoweekday()
        row = 1
        print("weekdat:%s!!!!!!!!!"%weekday)
        if weekday != 1:
            start = datetime(year, month, 1)
            pre_date = (start + timedelta(days=-(weekday - 1))).day
            for pre_day in range(1, weekday):
                draw_day(row, pre_day, pre_date, 0)
                pre_date += 1

        for day in range(1, total_day + 1):
            date = datetime.date(datetime(year=year, month=month, day=day))
            weekday = date.isoweekday()
            if weekday == 1 and day != 1: row += 1
            draw_day(row, weekday, day, 1)

        post_date = datetime(year, month, total_day)

        for post_day in range(7 - weekday + 7):
            post_date = post_date + timedelta(days=1)
            post_weekday = post_date.isoweekday()
            print(post_weekday)
            if post_weekday == 1: row += 1
            if row==7:
                break
            draw_day(row, post_weekday, post_date.day, 0)

    draw_calendar(year, month)
    print('start to puttext')
    img = puttext_list(img, text, 50)
    print('start to draw stage')
    processes=[100,100,100]
    
    
    processes.append(show_rate)
    draw_stage(processes, time=pre_time,img_list=latest_floors)
    print('start to draw main down')
    print(stage,percentage)
    img = draw_main_down(stage, percentage)
    # puttext(img, "hello", (100, 100), (255, 255, 255), 25)

    path1 = path_photo
    print('start to load image')
    img_main_pic = cv2.imread(path1)
    img_main_pic = cv2.resize(img_main_pic, (670, 434))
    print(img_main_pic.shape)

    img[231:665, 851:1521, ] = img_main_pic

    img = cv2.resize(img, (2580, 1280))
    img=img[86*2:623*2,60*2:1050*2,:]
    img = cv2.resize(img, (1920, 1080))
    cv2.imwrite(save_path, img)
    print(img.shape)
    return save_path

# if __name__=='__main__':
#     img1=cv2.imread('../current.jpg')
#     print(img1.shape)
#     img1=img1[86:623,60:1050,:]
#     print(img1.shape)

#     cv2.imwrite('./test_img.jpg',img1)



#draw(2022,8,path_photo='test_img.jpg',floors=[6,7],day_list=[10, 6, 7, 7, 3],save_path='result.png',percentage=15,stage=3,latest_floors=[],pre_time=['2022-8','2022-9'])
# -*- coding: utf-8 -*-
import os
import time
from controlAndSnapshot import *
from download import listSnapshots, download_snapshots
from stich import stitch



"""
    https://developer.qiniu.com/qvs/api/6739/enable-the-flow
    :param access_key: 公钥
    :param secret_key: 私钥
    :param namespaceId: 空间ID
    :param streamId: 流ID
    :param gbId: 设备国标Id
    :return:
            {
                "code": 200
            }
"""


class Camera:
    def __init__(self, access_key, secret_key, namespaceId, streamId, gbId):
        self.access_key = access_key
        self.secret_key = secret_key
        # 空间ID
        self.namespaceId = namespaceId
        # 流ID
        self.streamId = streamId
        # 设备国标Id
        self.gbId = gbId

    def prepare(self):
        # 启动流
        headers, result = enableStreams(self.access_key, self.secret_key, self.namespaceId, self.streamId)
        print("【启动流】 " + str(json.loads(headers)['code']))

        # 启动设备拉流
        headers, result = startDevice(self.access_key, self.secret_key, self.namespaceId, self.gbId)
        print("【启动设备拉流】 " + str(json.loads(headers)['code']))
        time.sleep(3)

        #
        # print(self.getPresetBit())
        #
        # 回到预置位
        # headers, result = controlPresetBit(self.access_key, self.secret_key, self.namespaceId, self.gbId, {
        #     'cmd': 'goto',
        #     # 'name': "pos0",
        #     'presetId': "1"
        # })
        # print("【回到预制位】 " + str(json.loads(headers)['code']))
        # time.sleep(4)
        # #
        # # headers, result = takeScreenshot(self.access_key, self.secret_key, self.namespaceId, self.streamId)
        # print("【截图】 " + str(json.loads(headers)['code']))

    def setPresetBit(self, name):
        headers, result = controlPresetBit(self.access_key, self.secret_key, self.namespaceId, self.gbId, {
            'cmd': 'set',
            'name': name,
            # 'presetId': "1"
        })
        print(headers)
        return json.loads(headers)['code']

    def getPresetBit(self):
        headers, result = listPresets(self.access_key, self.secret_key, self.namespaceId, self.gbId)
        print(headers, result)
        return json.loads(headers)

    def pic(self):
        headers, result = takeScreenshot(self.access_key, self.secret_key, self.namespaceId, self.streamId)
        print("【截图】 " + str(json.loads(headers)['code']))
        time.sleep(1)

    def move(self, cmd, speed, t):
        headers, result = controlConsole(self.access_key, self.secret_key, self.namespaceId, self.gbId, {
            'cmd': cmd,
            'speed': speed
        })
        print("【云台控制：" + cmd + "】 " + str(json.loads(headers)['code']))
        time.sleep(t)

        headers, result = controlConsole(self.access_key, self.secret_key, self.namespaceId, self.gbId, {
            'cmd': "stop",
        })
        print("【停止云台控制操作】 " + str(json.loads(headers)['code']))

        time.sleep(15)

    def moveAndPic(self, cmd, speed, t):
        self.move(cmd, speed, t)
        self.pic()

    def stop(self):
        # 停止设备拉流
        headers, result = stopDevice(self.access_key, self.secret_key, self.namespaceId, self.gbId)
        print("【停止设备拉流】 " + str(json.loads(headers)['code']))
        # 禁用流
        headers, result = stopStreams(self.access_key, self.secret_key, self.namespaceId, self.gbId)
        print("【禁用流】 " + str(json.loads(headers)['code']))

    # 查询设备信息
    def info(self):
        headers, result = listNamespacesInfo(self.access_key, self.secret_key, self.namespaceId, self.gbId)
        return result

    def snapshot(self):
        ################## 操作云台并截图 ########################
        start_time = int(time.time())

        self.prepare()

        # self.move("up", 5, 6)
        # self.move("zoomout", 5, 2)
        # self.move("right", 5, 15)
        print("!!!!!!!!!!!!!!!!!!!!")
        time.sleep(5.5)
        self.pic()
        time.sleep(5.5)
        self.pic()
        time.sleep(3.5)
        self.pic()

        # self.move("right", 10, 15)

        # self.move("left", 5, 1)
        # for i in range(1):
        #     print("!!!!!!!!!!!!!!!!!!!!")
        #     # 相机云台控制
        #     self.moveAndPic("left", 5, 1)
        #     self.moveAndPic("up", 3, 1)
        # self.move("down", 3, 1)

        self.stop()

        finish_time = int(time.time())
        time.sleep(7)

        ################## 下载截图并拼接 ########################
        code, res = listSnapshots(self.access_key, self.secret_key, self.namespaceId, self.streamId,
                                  100, None, start_time, finish_time)

        if code!=200:
            time.sleep(5)
            print('trying again...')
            code, res = listSnapshots(self.access_key, self.secret_key, self.namespaceId, self.streamId,
                                  100, None, start_time, finish_time)

        print(code,111)
        if code == 200:
            file_prefix = f"{os.getcwd()}/snapshots/{self.streamId}/{start_time}"
            print(file_prefix)
            try:
                os.makedirs(file_prefix)
            except Exception as e:
                print(e)
                pass
                

            print("【截图列表获取成功】")
            for image in res.get("items"):
                filename = time.strftime("20%y%m%d%H%M%S", time.localtime(image.get("time"))) + ".jpg"
                download_snapshots(self.access_key, self.secret_key, image.get("snap"), f"{os.path.join(file_prefix, filename)}")

            output_prefix = f"{os.getcwd()}/pics/{self.streamId}"
            if not os.path.isdir(output_prefix):
                os.makedirs(output_prefix)
            if not os.path.isdir(file_prefix):
                os.makedirs(file_prefix)
            stitch(f"{file_prefix}", 0,
                   f'{output_prefix}/{time.strftime("20%y%m%d%H%M%S", time.localtime(start_time))}.jpg')
            print('finish stitching')
        else: print('error')
        return f'{output_prefix}/{time.strftime("20%y%m%d%H%M%S", time.localtime(start_time))}.jpg'

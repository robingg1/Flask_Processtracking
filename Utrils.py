from camera import Camera

#from main_2 import main

def snapshot(access_key,secret_key,namespaceId,streamId,gbId):
    camera = Camera(access_key, secret_key, namespaceId, streamId, gbId)
    path = camera.snapshot()
    return path

if __name__ == '__main__':
    access_key = 'pGWOH5rkQFmvfIO1PxSLkTphLAyHtcbC-y8W5bbL'
    secret_key = 'Eku1QAiJTrVRjrTkwcqEvUXNKRGAIt0ednYHq-1O'
    # # 空间ID
    namespaceId = "jiji"
    # 流ID
    # streamId = "31011500991320014931"
    # # 设备国标Id
    # gbId = "31011500991320014931"
    # 流ID
    streamId = "31011500991320015266"
    # 设备国标Id
    gbId = "31011500991320015266"

    camera = Camera(access_key, secret_key, namespaceId, streamId, gbId)
    path=camera.snapshot()

    #main(path)

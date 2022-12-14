from qiniu.services.qvs.camera import Camera

if __name__ == '__main__':
    access_key = 'pGWOH5rkQFmvfIO1PxSLkTphLAyHtcbC-y8W5bbL'
    secret_key = 'Eku1QAiJTrVRjrTkwcqEvUXNKRGAIt0ednYHq-1O'
    # # 空间ID
    namespaceId = "jiji"
    # 流ID
    # streamId = "31011500991320014931"

    # gbId = "31011500991320014931"
    # 流ID
    # streamId = "31011500991320015266"
    # 设备国标Id
    # gbId = "31011500991320015266"

    # streamId = "31011500991320016434"
    # gbId = "31011500991320016434"

    streamId = "31011500991320016917"
    gbId = "31011500991320016917"

    camera = Camera(access_key, secret_key, namespaceId, streamId, gbId)
    camera.prepare()
    camera.move("up", 5, 1)

    camera.snapshot()

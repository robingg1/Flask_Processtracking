from main_2 import evaluate
from show.main import draw
import time
import base64
from Utrils import snapshot
import json
import datetime
import cv2
import requests
from flask import Flask, request, jsonify
from dateutil import rrule

app = Flask(__name__)


process = 0
REMOTE_SETTING_PATH = "setting.json"
LOCAL_SETTING_PATH = "config.json"


@app.route('/access')
def hello_world():
    # 测试flask是否正常工作
    return 'Hello Surface!'


@app.route('/setting', methods=["PUT"])
def set_setting():
    try:
        remote_setting_json = request.json

        local_setting_json = {"floor": [remote_setting_json["project"]["initial_floor"]], "days": [3], "pre_imgs": [], "pre_finish_t": [],
                              "percentage": remote_setting_json["project"]["initial_phase_presentage"],
                              "start_date": remote_setting_json["project"]["initial_date"],
                              "phase": remote_setting_json["project"]["initial_phase"],
                              "past_3_res": [], }
        # 'pre_res'=[]}

        with open(REMOTE_SETTING_PATH, 'w') as fid:
            json.dump(remote_setting_json, fid)

        with open(LOCAL_SETTING_PATH, 'w') as fid:
            json.dump(local_setting_json, fid)

        return_value = {
            "is_successful": True,
            "msg": "设置成功！"
        }
    except Exception as e:
        print(e)
        return_value = {
            "is_successful": False,
            "msg": "error happen for json save"
        }
    return jsonify(return_value)


def calculate(str1):
    time1 = datetime.datetime.today()
    year = time1.year
    month = time1.month
    day = time1.day
    # time1=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
    print(time1, year, month, day)

    str1 = str1.split("-")
    year1 = int(str1[0])
    month1 = int(str1[1])
    day1 = int(str1[2])
    last_day = datetime.datetime(year, month, day)
    first_day = datetime.datetime(year1, month1, day1)

    todayy = f'{year}-{month}-{day}'
    days = last_day-first_day

    # days = month * 30 + day - (month1 * 30 + day1)
    # days=(month-month1)*30+day-day1
    # days = rrule.rrule(freq = rrule.DAILY,dtstart=first_day,until=last_day)
    # print(days.counts())
    return int(days.days), todayy


@app.route('/setting', methods=['get'])
def get_setting():
    try:
        try:
            with open(REMOTE_SETTING_PATH, 'r') as file:
                remote_setting_json = json.load(file)
        except Exception as e:
            print('config file not setted yet')
        return_value = {
            "is_successful": True,
            "msg": "OK",
            "data": remote_setting_json
        }
    except Exception as e:
        return_value = {
            "is_successful": False,
            "msg": "ERROR",
        }
        print('error')

    return jsonify(return_value)


def take_photo_eval(pic_only, session_id):
    global process

    try:
        with open(REMOTE_SETTING_PATH, 'r') as file:
            remote_setting_json = json.load(file)
    except:
        print('error')

    try:
        with open(LOCAL_SETTING_PATH, 'r') as file:
            local_setting_json = json.load(file)
    except:
        print('error')

    # print(data2)

    segmentation_limit = remote_setting_json["segmentation"]
    classification_limit = remote_setting_json["classification"]

    percentage = local_setting_json["percentage"]
    start_time = local_setting_json["start_date"]
    past_3_res = local_setting_json["past_3_res"]
    # pre_res=local_setting_json['pre_res']
    print(f'past percentage is {percentage}, past start_time is {start_time}')

    process = 0.1
    camera_setting = remote_setting_json["qiniu_setting"]

    access_key = camera_setting["app_key"]
    secret_key = camera_setting["app_secret"]
    namespaceId = camera_setting["namespaceId"]
    streamId = camera_setting["streamId"]
    gbId = camera_setting["streamId"]
    pic_path = remote_setting_json["callback_pic"]
    res_path = remote_setting_json["callback_result"]

    print(access_key, secret_key, namespaceId, gbId)
    try:
        path = snapshot(access_key, secret_key, namespaceId, streamId, gbId)
 
    #path = '/usr/local/work_surface/flaskProject/pics/31011500991320016917/20221014093000.jpg'
        print(path)

        with open(path, 'rb') as f:
            img_data = f.read()
            ibase64_data = base64.b64encode(img_data)
            base64_str = str(ibase64_data, 'utf-8')

        data_pic = {}
        print('start to callback')

        data_pic["pic_base64"] = 'data:image/jpeg;base64,' + base64_str
        data_pic["time"] = int(time.time())
        print(int(time.time()))
        data_pic["session_id"] = session_id
        try:
            r = requests.post(pic_path, headers={
                            'Content-Type': 'application/json'}, timeout=10, json=data_pic).json()
        except Exception as ex:
            pass
        process = 0.3

        if pic_only != 'True':
            data_f = {}
            res1 = {}
            # evaluate for img
            try:
                stage, p_res, res = evaluate(path)
                past_3_res.append(res)
                if len(past_3_res) > 3:
                    past_3_res.pop(0)

                process = 0.5
                res1['phase'] = int(stage)
                res1["phase_percentage"] = float(p_res)
                res1["global_percentage"] = float(res)

                print(f'current stage is {stage}, percentage:{res}')
                duration, today = calculate(start_time)
                show_rate = round(res, 2)

                if (len(local_setting_json["pre_imgs"]) == 0):
                    local_setting_json["pre_imgs"].append(path)
                if (len(local_setting_json["pre_finish_t"]) == 0):
                    local_setting_json["pre_finish_t"].append(today)

                if len(past_3_res) == 3 and past_3_res[0] >= 0.55 and past_3_res[1] < 0.2 and past_3_res[2] < 0.2:
                    print('update new stairs')
                    time1 = time.time()
                    # duration, today = calculate(start_time)
                    #    duration=time1-start_time
                    local_setting_json["start_date"] = today
                    local_setting_json["floor"].append(
                        local_setting_json["floor"][-1] + 1)
                    local_setting_json["phase"] = 1
                    local_setting_json["percentage"] = 0.01
                    local_setting_json["past_3_res"] = []
                    local_setting_json["days"][-1] = (duration)
                    local_setting_json["days"].append(1)
                    # local_setting_json["pre_imgs"].append(path)  # 当前图片
                    if local_setting_json["pre_imgs"][-1] != path:
                        local_setting_json["pre_imgs"].append(path)

                    if local_setting_json["pre_finish_t"][-1] != today:
                        local_setting_json["pre_finish_t"].append(today)

                elif len(past_3_res) <= 1 or past_3_res[-2] != res:
                    # if len(pre_res)<3:
                    #     res=max(percentage+0.003,res)
                    # else:

                    # local_setting_json["percentage"].append(res)
                    res = res
                    # if res <= percentage:
                    #     res = percentage + 0.03
                    res1["global_percentage"] = float(res)
                    # duration, today = calculate(start_time)
                    local_setting_json["days"][-1] = duration
                    local_setting_json["past_3_res"] = past_3_res
                    local_setting_json["percentage"] = round(res, 2)
                    if len(past_3_res) >= 2 and past_3_res[-2] > 0.7 and past_3_res[-1] < 0.2:
                        show_rate = 0.96
                    else:
                        show_rate = round(res, 2)

                    local_setting_json["phase"] = int(stage)
                    print(local_setting_json)
                process = 0.7
                with open(LOCAL_SETTING_PATH, 'w') as file:
                    file.write(json.dumps(local_setting_json))
            except Exception as e:
                print(e)
                res = 30
            print(stage, percentage)
            print(show_rate, 1111)

            draw(datetime.datetime.today().year, datetime.datetime.today().month, path, floors=local_setting_json["floor"],
                day_list=local_setting_json["days"], stage=stage, save_path='current.jpg',
                percentage=int(res*100), latest_floors=local_setting_json["pre_imgs"], pre_time=local_setting_json["pre_finish_t"], show_rate=int(show_rate*100))
            process = 0.9
            with open('current.jpg', 'rb') as f:
                img_data = f.read()
                ibase64_data = base64.b64encode(img_data)
                base64_str = str(ibase64_data, 'utf-8')

            data_f["time"] = int(time.time())
            data_f["session_id"] = session_id
            data_f["pic_base64"] = 'data:image/jpeg;base64,' + base64_str
            data_f['result'] = res1

            # post to callback_pic
            try:
                r = requests.post(res_path, headers={
                                'Content-Type': 'application/json'}, timeout=10, json=data_f).json()
                print(r)
            except Exception as ex:
                print(ex)
                print('回传错误')
                pass
            process = 0
    except:
            process = 0


@app.route('/start', methods=["get"])
def take_photo_and_evaluate():
    try:
        global process

        if request.method == 'GET':

            session_id = request.values.get('session_id')
            pic_only = request.values.get("pic_only")
            url = request.host_url
            print(f'url is {url}')
            print(type(pic_only))
            print(f'session_id is {session_id}, pic_only is {pic_only}')

            req2 = requests.get(url='http://127.0.0.1:8985/status')
            print(req2.json())
            if req2.json()['msg'] == 'busy':
                return_data = {
                    "is_successful": False,
                    "msg": "服务器正在工作，请不要重复执行。"
                }
                return jsonify(return_data)
            else:
                try:
                    from threading import Thread
                    p = Thread(target=take_photo_eval,
                               args=(pic_only, session_id,))
                    p.daemon = True
                    p.start()
                    return_data = {
                        "is_successful": True,
                        "msg": "识别已启动，需要等待10分钟后查看结果。"
                    }
                except Exception as e:
                    print(e)
                    return_data = {
                        "is_successful": False,
                        "msg": "evaluation error"
                    }

        else:
            return_data = {
                "is_successful": False,
                "msg": "Only GET method is supported"
            }

    except Exception as e:
        print(e)
        return_data = {
            "is_successful": False,
            "msg": "Evaluation error"
        }

    return jsonify(return_data)


@app.route('/status', methods=["GET"])
def status():
    global process
    try:
        state = "busy"
        if process == 0:
            state = 'idle'
        data = {
            "is_successful": True,
            "msg": state,
            "progress": process

        }
    except Exception as e:
        data = {
            "is_successful": False,
            "msg": 'Error',
            "progress": process

        }
    return jsonify(data)


if __name__ == '__main__':
    # calculate('2022-8-11')
    app.run(host='0.0.0.0', port=8985)

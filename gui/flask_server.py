import imp
import json
import os
import sys
import time

import pyaudio
from flask import Flask, render_template, request, Response
from flask_cors import CORS

import fay_booter
from core.interact import Interact
from core.question_db import question_db
from core.shuziren_db import shuziren_db

from core.tts_voice import EnumVoice
from gevent import pywsgi
from scheduler.thread_manager import MyThread
from utils import config_util, util
from core import wsa_server
from core import fay_core
from core.content_db import Content_Db

# from ai_module import yolov8


BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
print("当前的根目录为：", BASE_DIR)
__app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'gui/templates'), static_folder=os.path.join(BASE_DIR, 'gui/static'))
__app.templates_auto_reload=True
# __app.config['TEMPLATES_DIR'] = 'D:\\templates'
CORS(__app, supports_credentials=True)


def __get_template():
    return render_template('index.html')


def __get_device_list():
    audio = pyaudio.PyAudio()
    device_list = []
    for i in range(audio.get_device_count()):
        devInfo = audio.get_device_info_by_index(i)
        if devInfo['hostApi'] == 0:
            device_list.append(devInfo["name"])

    return list(set(device_list))


@__app.route('/api/submit', methods=['post'])
def api_submit():
    data = request.values.get('data')
    # print(data)
    config_data = json.loads(data)
    if(config_data['config']['source']['record']['enabled']):
        config_data['config']['source']['record']['channels'] = 0
        audio = pyaudio.PyAudio()
        for i in range(audio.get_device_count()):
            devInfo = audio.get_device_info_by_index(i)
            if devInfo['name'].find(config_data['config']['source']['record']['device']) >= 0 and devInfo['hostApi'] == 0:
                 config_data['config']['source']['record']['channels'] = devInfo['maxInputChannels']

    # BRONCOS: 这里的目的是禁止前端修改["interact"]["playSound"]的值，但此处可能没有也可以，需要测试一下
    #           但最好还是带着，逻辑不出错
    config_data['config']["interact"]["playSound"] = config_util.config["interact"]["playSound"]


    config_util.save_config(config_data['config'])
    return '{"result":"successful"}'

# @__app.route('/api/control-eyes', methods=['post'])
# def control_eyes():
#     eyes = yolov8.new_instance()
#     if(not eyes.get_status()):
#        eyes.start()
#        util.log(1, "YOLO v8正在启动...")
#     else:
#        eyes.stop()
#        util.log(1, "YOLO v8正在关闭...")
#     return '{"result":"successful"}'


@__app.route('/api/get-data', methods=['post'])
def api_get_data():
    wsa_server.get_web_instance().add_cmd({
        "voiceList": [
            {"id": EnumVoice.XIAO_XIAO.name, "name": "晓晓"},
            {"id": EnumVoice.YUN_XI.name, "name": "云溪"}
        ]
    })
    wsa_server.get_web_instance().add_cmd({"deviceList": __get_device_list()})
    return json.dumps({'config': config_util.config})


@__app.route('/api/start-live', methods=['post'])
def api_start_live():
    # time.sleep(5)
    fay_booter.start()
    time.sleep(1)
    wsa_server.get_web_instance().add_cmd({"liveState": 1})
    return '{"result":"successful"}'


@__app.route('/api/stop-live', methods=['post'])
def api_stop_live():
    # time.sleep(1)
    fay_booter.stop()
    time.sleep(1)
    wsa_server.get_web_instance().add_cmd({"liveState": 0})
    return '{"result":"successful"}'

@__app.route('/api/send', methods=['post'])
def api_send():
    data = request.values.get('data')
    info = json.loads(data)
    text = fay_core.send_for_answer(info['msg'],info['sendto'])
    return '{"result":"successful","msg":"'+text+'"}'

@__app.route('/api/add_question', methods=['post'])
def api_add_question():
    data = request.values.get('data')
    info = json.loads(data)
    text = question_db.add_content(info['question'], info['answer'])
    util.config_util.video_stream.window.update_signal.emit("")
    return '{"result":"successful","msg":"' + str(text) + '"}'
@__app.route('/api/add_shuziren', methods=['post'])
def api_add_shuziren():
    data = request.values.get('data')
    info = json.loads(data)
    text = shuziren_db.add_content(info['name'], info['sound'], info['image'] if 'image' in info else '', info['video'] if 'video' in info else '',info['video2'] if 'video2' in info else '')
    util.config_util.video_stream.window.update_signal_shuziren.emit("")
    return '{"result":"successful","msg":"' + str(text) + '"}'

@__app.route('/api/upload_file', methods=['post'])
def upload_file():
    if 'file' in request.files:
        video = request.files['file']
        if video.filename != '':
            # 确保文件名安全，避免潜在的安全风险
            filename = video.filename
            video.save(os.path.join("./gui/static", filename))
            return '{"result":"successful","msg":"' + '/static/'+filename +'","raw":"'+filename+ '"}'
    return '{"result":"error","msg":"'  + '"}'

@__app.route('/api/speak', methods=['post'])
def api_speak():
    try:
        data = request.get_json()
        msg = data["msg"]
        feiFei = fay_booter.feiFei
        if len(msg) > 0:
            feiFei.change_muting(False)
            interact = Interact("mic", 1, {'user': '', 'msg': msg})
            # 记录日志信息
            util.printInfo(3, "视觉信息", '{}'.format(interact.data["msg"]), time.time())
            # 内容所在
            feiFei.on_interact(interact)
            return '{"result":"successful"}'
        else:
            return '{"result":"error"}'
    except Exception as e:
        print(e)
        return '{"result":"error"}'
    except BaseException as e:
        print(e)
        return '{"result":"error"}'
    # info = json.loads(data)
    # text = question_db.add_content(info['question'], info['answer'])
    # util.config_util.video_stream.window.update_signal.emit("")


@__app.route('/api/get-msg', methods=['post'])
def api_get_Msg():
    contentdb = Content_Db()
    list = contentdb.get_list('all','desc',1000)
    relist = []
    i = len(list)-1
    while i >= 0:
        relist.append(dict(type=list[i][0],way=list[i][1],content=list[i][2],createtime=list[i][3],timetext=list[i][4]))
        i -= 1

    return json.dumps({'list': relist})

@__app.route('/api/get-question-count', methods=['post'])
def api_get_question_count():
    list = question_db.get_count()

    return json.dumps({'count': list})
@__app.route('/api/get-question', methods=['post'])
def api_get_question():
    data = request.values.get('data')
    info = json.loads(data)

    list = question_db.get_list(info['n'],info['l'])
    count = question_db.get_count()
    return json.dumps({'list': list,'count': count})
@__app.route('/api/edite-question', methods=['post'])
def api_edite_question():
    data = request.values.get('data')
    info = json.loads(data)
    count = question_db.edite(info)
    util.config_util.video_stream.window.update_signal.emit("")
    return ""
@__app.route('/api/del-question', methods=['post'])
def api_del_question():
    data = request.values.get('data')
    count = question_db.delete(data)
    util.config_util.video_stream.window.update_signal.emit("")
    return ""

@__app.route('/api/get-shuziren-count', methods=['post'])
def api_get_shuziren_count():
    list = shuziren_db.get_count()

    return json.dumps({'count': list})
@__app.route('/api/get-shuziren', methods=['post'])
def api_get_shuziren():
    data = request.values.get('data')
    info = json.loads(data)

    list = shuziren_db.get_list(info['n'],info['l'])
    count = shuziren_db.get_count()
    return json.dumps({'list': list,'count': count})
@__app.route('/api/edite-shuziren', methods=['post'])
def api_edite_shuziren():
    data = request.values.get('data')
    info = json.loads(data)
    count = shuziren_db.edite(info)
    util.config_util.video_stream.window.update_signal_shuziren.emit("")
    return ""
@__app.route('/api/del-shuziren', methods=['post'])
def api_del_shuziren():
    data = request.values.get('data')
    count = shuziren_db.delete(data)
    util.config_util.video_stream.window.update_signal_shuziren.emit("")
    return ""

@__app.route('/', methods=['get'])
def home_get():
    return __get_template()


@__app.route('/', methods=['post'])
def home_post():
    return __get_template()

def event_stream():
    for i in range(20):
        time.sleep(1)  # 模拟事件生成的延迟
        print("发送",i)
        yield f"data: {json.dumps({'message': 'Hello, World!'})}\n\n"


@__app.route('/sse', methods=['POST'])
def sse():
    def generate():
        for event in event_stream():
            yield event
    return Response(generate(), mimetype='text/event-stream')

def run():
    server = pywsgi.WSGIServer(('0.0.0.0',5000), __app)
    server.serve_forever()

def start():
    MyThread(target=run).start()

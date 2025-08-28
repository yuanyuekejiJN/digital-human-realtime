import base64
import binascii
import json
import os
import socket
import time
import uuid
import webbrowser

import requests
import wmi
from Crypto.Cipher import AES
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication

from ai_module import ali_nls, ali_tts_sdk
from core import wsa_server, shuziren_db
from gui import flask_server
from gui.window import MainWindow
from utils import config_util
from core.content_db import Content_Db
import sys
from gui.video_window import VideoWindow, VideoStream

sys.setrecursionlimit(sys.getrecursionlimit() * 50000)
sys.setswitchinterval(sys.getswitchinterval() * 8000000)
file_path = "secret.bin"
key = b'abcdefghijklmnop'  # AES - 128密钥长度为16字节
print("------1>")
def __clear_samples():
    if not os.path.exists("./samples"):
        os.mkdir("./samples")
    for file_name in os.listdir('./samples'):
        if file_name.startswith('sample-'):
            os.remove('./samples/' + file_name)


def __clear_songs():
    if not os.path.exists("./songs"):
        os.mkdir("./songs")
    for file_name in os.listdir('./songs'):
        if file_name.endswith('.mp3'):
            os.remove('./songs/' + file_name)


def __clear_logs():
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    for file_name in os.listdir('./logs'):
        if file_name.endswith('.log'):
            os.remove('./logs/' + file_name)


# ws在线测试：https://docs.wildfirechat.cn/web/wstool/index.html
# TODO: 清空缓存功能，暂未实现
def __clear_cache():
    pass


def aes_decrypt(key, encrypted_data):
    aes = AES.new(key, AES.MODE_ECB)
    decrypted_data = aes.decrypt(base64.b64decode(encrypted_data))
    padding_len = decrypted_data[-1]
    return decrypted_data[0: - padding_len].decode('utf-8')




def get_windows_product_id():
    try:
        # s = wmi.WMI()
        # mainboard = []
        # for board_id in s.Win32_BaseBoard():
        #     mainboard.append(board_id.SerialNumber.strip().strip('.'))
        # return mainboard[0]
        s = wmi.WMI()
        cpus = s.Win32_Processor()
        mainboard = []
        for cpu in cpus:
            # print(f"CPU ID: {cpu.ProcessorId}")
            mainboard.append(cpu.ProcessorId.strip().strip('.'))
        # for board_id in s.Win32_BaseBoard():
        #     mainboard.append(board_id.SerialNumber.strip().strip('.'))
        return mainboard[0]

    except Exception as e:
        print("获取 Windows 产品 ID 出错:", e)
        return None

# 判断授权码
def check_key(content,device_id):
    global key
    try:
        decrypted_result = aes_decrypt(key, content)
        print("解密结果:", decrypted_result)
        info = json.loads(decrypted_result)
        if info['id'] == device_id and info["time"] > int(time.time()):
            print("授权通过")
            return True
        else:
            print("授权未通过")
            if info['id'] != device_id:
                print("设备未授权")
            elif info["time"] < int(time.time()):
                print("授权到期了")
            return False
    except BaseException as e:
        print(e)
        return False


#输入授权码
def input_key(device_id):
    global file_path
    while True:
        user_input = input("请输入授权码: ")
        with open(file_path, 'w') as file:
            file.write(user_input)
        if check_key(user_input, device_id):
            print("授权码正确，欢迎！")
            break
        else:
            print("授权码错误，请重新输入。请把设备号发送给元岳科技工作人员,获取授权码")

def main():
    global file_path
    global key
    device_id = get_windows_product_id()
    print(f"设备号:{device_id}")
    try:
        content = ""
        with open(file_path, 'r') as file:
            content = file.read()
            print(content)
        if not check_key(content, device_id):
            input_key(device_id)
    except FileNotFoundError:
        print("未授权请把设备号发送给元岳科技工作人员,获取授权码")
        input_key(device_id)
    # 清空目录
    __clear_samples()
    __clear_songs()
    __clear_logs()
    __clear_cache()

    # 载入配置文件
    config_util.load_config()

    # 数据库不存在就创建
    if not os.path.exists("fay.db"):
        contentdb = Content_Db()
        contentdb.init_db()
    shuziren_db.shuziren_db.init()

    # 初始化websocket: 10002为与UE5的端口, 10003为与web面板的端口
    ws_server = wsa_server.new_instance(port=10002)
    ws_server.start_server()
    web_ws_server = wsa_server.new_web_instance(port=10003)
    web_ws_server.start_server()

    # 根据配置启动语音服务
    if config_util.ASR_mode == "ali":
        ali_nls.start()

    if config_util.tts_mode == "ali":
        ali_tts_sdk.start()

    # 启动Flask服务器
    flask_server.start()

    # 启动PyQt应用程序
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('icon.png'))

    # MainWindow()

    video_window = VideoWindow()
    video_window.ui.show()

    video_stream = VideoStream(video_window)
    config_util.video_stream = video_stream
    video_stream.start()

    # 打开网址
    webbrowser.open('http://127.0.0.1:5000')

    sys.exit(app.exec_())

def main2():
    flask_server.start()

if __name__ == '__main__':
    main()
    # main2()
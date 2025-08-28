import base64
from collections import deque
from threading import Thread

import websocket
import json
import time
import ssl
import _thread as thread
import hashlib
import hmac
from urllib.parse import quote
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

from core import wsa_server, song_player
from scheduler.thread_manager import MyThread
from utils import util
from utils import config_util as cfg

from websocket import create_connection


__running = True
__my_thread = None


def start():
    pass

class XfNls:
    # 初始化
    def __init__(self, result=""):
        self.base_url = "ws://rtasr.xfyun.cn/v1/ws"
        self.app_id = "db838bdb"
        self.api_key = "d41b583e2a69220aeeb277ac8f2971f9"
        self.end_tag = "{\"end\": true}"

        self.__ws = None
        self.__connected = False
        self.__frames = deque()
        self.__state = 0
        self.__closing = False
        self.__task_id = ''
        self.done = False
        self.finalResults = result
        self.video_stream = util.config_util.video_stream

    # 获取签名 科大讯飞
    def get_sign(self):
        ts = str(int(time.time()))
        tt = (self.app_id + ts ).encode('utf-8')
        md5 = hashlib.md5()
        md5.update(tt)
        baseString = md5.hexdigest()
        baseString = bytes(baseString, encoding='utf-8')
        punc = '0'
        apiKey = self.api_key.encode('utf-8')
        signa = hmac.new(apiKey, baseString, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        res = "ts=" + ts + "&signa=" + quote(signa) + "&punc=" + punc
        # res = "ts=" + ts + "&signa=" + quote(signa)
        return res

    # BR: done
    def __on_msg(self):
        if "暂停" in self.finalResults or "不想听了" in self.finalResults or "别唱了" in self.finalResults:
            song_player.stop()

    # BR: done
    # 收到websocket消息的处理
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            # TODO: 这里要看一下格式是什么样的，有没有开翻译功能  ！！！需控制台开通翻译功能！！！

            msgAction = data['action']
            # msgCode = data['code']
            results = json.loads(data['data'])
            print(results)
            cn_value = results['cn']

            # 获取值中'st'键对应的值
            st_value = cn_value['st']
            resultss = ''
            # 遍历'st'值中'rt'列表中的每个元素
            for rt_item in st_value['rt']:
                # 在每个元素中遍历'ws'列表中的每个元素
                for ws_item in rt_item['ws']:
                    # 提取每个'cw'字典中'w'键对应的值
                    cw_value = ws_item['cw']
                    for cw_item in cw_value:
                        w_content = cw_item['w']
                        resultss += w_content
                        wb = cw_item['wb']
                        # ls = results['ls']
            print(resultss)
            # msgData = {"ls": ls,"src": resultss}
            msgData = {"wb": wb,"src": resultss}
            if msgAction == 'result':
                print("执行到这里了")
                # if msgData['ls']:
                #     self.done = True
                if msgData['wb'] != 0:
                    self.done = True
                self.finalResults = msgData['src']
                wsa_server.get_web_instance().add_cmd({"panelMsg": self.finalResults})

                # BRONCOS--QUESTION 标记
                self.video_stream.window.answer_signal.emit("")
                self.video_stream.window.question_signal.emit(self.finalResults)  # 更新问题

                if not cfg.config["interact"]["playSound"]: # 非展板播放
                    # 这里指的是一个字一个字的传输的内容
                    content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': self.finalResults}}   # 发送文字的地方
                    wsa_server.get_instance().add_cmd(content)
                self.__on_msg()

        except Exception as e:
            print(e)
        # print("### message:", message)
        if self.__closing:
            try:
                self.__ws.close()
            except Exception as e:
                print(e)

    # 收到websocket错误的处理
    def on_close(self, ws, code, msg):
        self.__connected = False
        print("### CLOSE:", msg)

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        print("### error:", error)

    # 收到websocket连接建立的处理
    def on_open(self, ws):
        # 连接上ws后会自动执行当前的回调函数
        self.__connected = True

        # print("连接上了！！！")

        def run(*args):
            while self.__connected:
                try:
                    if len(self.__frames) > 0:
                        frame = self.__frames.popleft()
                        if type(frame) == dict:
                            ws.send(json.dumps(frame))
                        elif type(frame) == bytes:
                            ws.send(frame, websocket.ABNF.OPCODE_BINARY)
                        # print('DEBUG: [ali-nls, 136] 发送 ------> ' + str(type(frame)))
                except Exception as e:
                    print(e)
                time.sleep(0.04)

        # 单独开一个线程执行上面的run子方法
        thread.start_new_thread(run, ())

    # BR: done
    def __connect(self):
        self.finalResults = ""
        self.done = False
        self.__frames.clear()
        self.__ws = websocket.WebSocketApp(self.base_url + "?appid=" + self.app_id + "&" + self.get_sign(), on_message=self.on_message)
        self.__ws.on_open = self.on_open
        self.__ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    # BR: done
    def add_frame(self, frame):
        self.__frames.append(frame)

    # BR: done
    def send(self, buf):
        self.__frames.append(buf)

    # BR: done
    def start(self):
        Thread(target=self.__connect, args=[]).start()

    # BR: done
    def end(self):
        if self.__connected:
            try:
                for frame in self.__frames:
                    self.__frames.popleft()
                    if type(frame) == dict:
                        self.__ws.send(json.dumps(frame))
                    elif type(frame) == bytes:
                        self.__ws.send(frame, websocket.ABNF.OPCODE_BINARY)
                    time.sleep(0.4)
                self.__frames.clear()
                self.__ws.send(bytes(self.end_tag.encode('utf-8')))
            except Exception as e:
                print(e)
        self.__closing = True
        self.__connected = False

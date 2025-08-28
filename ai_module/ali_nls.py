import difflib
import wave
from collections import deque
from threading import Thread, Lock

import websocket
import json
import time
import ssl
import _thread as thread
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

from core import wsa_server, song_player
from scheduler.thread_manager import MyThread
from utils import util, config_util
from utils import config_util as cfg
from wav2lip.utils import lazy_pinyin

__running = True
__my_thread = None
_token = ''


def __post_token():
    global _token
    __client = AcsClient(
        cfg.key_ali_nls_key_id,
        cfg.key_ali_nls_key_secret,
        "cn-shanghai"
    )

    __request = CommonRequest()
    __request.set_method('POST')
    __request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    __request.set_version('2019-02-28')
    __request.set_action_name('CreateToken')
    _token = json.loads(__client.do_action_with_exception(__request))['Token']['Id']


def __runnable():
    while __running:
        __post_token()
        time.sleep(60 * 60 * 12)


def start():
    MyThread(target=__runnable).start()


class ALiNls:
    # 初始化
    def __init__(self, result="",audio_id=""):
        self.__URL = 'wss://nls-gateway-cn-shenzhen.aliyuncs.com/ws/v1'
        self.__ws = None
        self.__connected = False
        self.__frames = []
        self.__state = 0
        self.__is_close = False
        self.__task_id = ''
        self.done = False
        self.finalResults = result
        self.video_stream = util.config_util.video_stream
        self.lock = Lock()
        self.started = False
        self.data = b''
        self.audio_id = audio_id



    def __create_header(self, name):
        if name == 'StartTranscription':
            self.__task_id = util.random_hex(32)
        header = {
            "appkey": cfg.key_ali_nls_app_key,
            "message_id": util.random_hex(32),
            "task_id": self.__task_id,
            "namespace": "SpeechTranscriber",
            "name": name
        }
        return header

    def __on_msg(self):
        pass
        # if "暂停" in self.finalResults or "不想听了" in self.finalResults or "别唱了" in self.finalResults:
        #     song_player.stop()

    # 收到websocket消息的处理
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            header = data['header']
            name = header['name']
            print('alinls-->',name,data)
            if name == 'SentenceEnd':
                self.finalResults += data['payload']['result']
                # self.done = True
                print('alinls-msg->', self.finalResults)
                # if (
                #         # (cfg.config["attribute"]["chufachi_r"] in ''.join(lazy_pinyin(self.finalResults))) or
                #         # # (difflib.SequenceMatcher(None, self.finalResults,cfg.config["attribute"]["huanxingchi"]).quick_ratio() > 0.6)
                #         # cfg.config["attribute"]["huanxingchi_r"] in ''.join(lazy_pinyin(self.finalResults))
                # ):
                self.video_stream.window.question_signal.emit(self.finalResults)
                if (
                        # (cfg.config["attribute"]["chufachi"] in lazy_pinyin(self.finalResults)) or
                        # (difflib.SequenceMatcher(None, self.finalResults,cfg.config["attribute"]["huanxingchi"]).quick_ratio() > 0.6)
                        cfg.config["attribute"]["open_huanxingchi"] and cfg.config["attribute"]["huanxingchi_r"] in ''.join(lazy_pinyin(self.finalResults))
                ):
                    config_util.set_last_question("")
                    # self.video_stream.window.pause_signal.emit(True)
                    self.video_stream.window.question_signal.emit(self.finalResults)

                if '电小月' in self.finalResults:
                    self.finalResults = self.finalResults.replace('电小月', '电小岳')
                wsa_server.get_web_instance().add_cmd({"panelMsg": self.finalResults})

                # BRONCOS--QUESTION 标记
                # self.video_stream.window.answer_signal.emit("")
                # self.video_stream.window.question_signal.emit(self.finalResults)  # 更新问题

                if not cfg.config["interact"]["playSound"]:# and cfg.config["attribute"]["muting"]: # 非展板播放
                    # 这里指的是一个字一个字的传输的内容
                    content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': self.finalResults}}
                    wsa_server.get_instance().add_cmd(content)
                self.__on_msg()
                # if self.__closing:
                # try:
                #     self.__ws.close()
                #     print("ali-ws-close-->")
                # except Exception as e:
                #     print("ali-ws-close-error->", e)
            elif name == 'TranscriptionResultChanged':
                self.finalResults2 = data['payload']['result']
                # self.finalResults = data['payload']['result']
                print('alinls-msg-c->', self.finalResults2)
                # if (
                #         (cfg.config["attribute"]["chufachi_r"] in ''.join(lazy_pinyin(self.finalResults))) or
                #         # (difflib.SequenceMatcher(None, self.finalResults,cfg.config["attribute"]["huanxingchi"]).quick_ratio() > 0.6)
                #         cfg.config["attribute"]["huanxingchi_r"] in ''.join(lazy_pinyin(self.finalResults))
                # ):
                    # self.video_stream.window.pause_signal.emit(True)
                self.video_stream.window.question_signal.emit(self.finalResults2)


                # self.finalResults = data['payload']['result']
                # print('alinls-msg->', self.finalResults)
                # if (
                #         # (cfg.config["attribute"]["chufachi"] in lazy_pinyin(self.finalResults)) or
                #         # (difflib.SequenceMatcher(None, self.finalResults,cfg.config["attribute"]["huanxingchi"]).quick_ratio() > 0.6)
                #         cfg.config["attribute"]["huanxingchi_r"] in ''.join(lazy_pinyin(self.finalResults))
                # ):
                #     self.video_stream.window.pause_signal.emit(True)
                #     self.video_stream.window.question_signal.emit(self.finalResults)
                #
                # wsa_server.get_web_instance().add_cmd({"panelMsg": self.finalResults})
                #
                # # BRONCOS--QUESTION 标记
                # # self.video_stream.window.answer_signal.emit("")
                # # self.video_stream.window.question_signal.emit(self.finalResults)  # 更新问题
                #
                # if not cfg.config["interact"]["playSound"]: # 非展板播放
                #     # 这里指的是一个字一个字的传输的内容
                #     content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': self.finalResults}}   # 发送文字的地方
                #     wsa_server.get_instance().add_cmd(content)
                # self.__on_msg()
            elif name == 'TranscriptionCompleted':
                try:
                    self.done = True
                    self.__ws.close()
                    print("ali-ws-close-->")
                except Exception as e:
                    print("ali-ws-close-error->", e)
            elif name == 'TranscriptionStarted':
                self.started = True

        except Exception as e:
            print("ali-on_message-error",e)
        # print("### message:", message)
        # if self.__closing:
        #     try:
        #         self.__ws.close()
        #         print("ali-ws-close-->")
        #     except Exception as e:
        #         print("ali-ws-close-error->",e)

    # 收到websocket错误的处理
    def on_close(self, ws, code, msg):
        self.__connected = False
        self.__is_close = True
        print("websocket### CLOSE:", msg)

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        print("websocket### error:", error)

    # 收到websocket连接建立的处理
    def on_open(self, ws):
        # 连接上ws后会自动执行当前的回调函数
        self.__connected = True

        print("ali-on_open->")

        def run(*args):
            try:
                while self.__connected:
                    try:
                        if len(self.__frames) > 0:
                            with self.lock:
                                frame = self.__frames.pop(0)
                            if isinstance(frame, dict):
                                # res = ws.send(json.dumps(frame))
                                if ws.sock:
                                    res = ws.sock.send(json.dumps(frame), websocket.ABNF.OPCODE_TEXT)
                                    print("ali-send->",res,json.dumps(frame))
                                else:
                                    print("ali-send-error->Connection is already closed",)
                            elif isinstance(frame, bytes):
                                while not self.started:
                                    time.sleep(0.01)
                                if ws.sock:
                                    res = ws.sock.send(frame, websocket.ABNF.OPCODE_BINARY)
                                    # print("ali-send-f->",res,"<->",len(frame))
                                    self.data += frame
                                else:
                                    print("ali-send-f-error->Connection is already closed",)
                        else:
                            time.sleep(0.001)  # 避免忙等
                    except Exception as e:
                        print("ali-send-error1->", e)
                        break

                if self.__is_close == False:
                    while len(self.__frames) > 0:
                        with self.lock:
                            frame = self.__frames.pop(0)
                        if isinstance(frame, dict):
                            # res = ws.send(json.dumps(frame))
                            if ws.sock:
                                res = ws.sock.send(json.dumps(frame), websocket.ABNF.OPCODE_TEXT)
                                print("ali-send2->", res,json.dumps(frame))
                            else:
                                print("ali-send2-error->Connection is already closed", )
                        elif isinstance(frame, bytes):
                            if ws.sock:
                                res = ws.sock.send(frame, websocket.ABNF.OPCODE_BINARY)
                                print("ali-send2-f->", res, "<->", len(frame))
                                self.data += frame
                            else:
                                print("ali-send2-f-error->Connection is already closed", )
                    print("ali-send3->", len(self.__frames))
                    frame = {"header": self.__create_header('StopTranscription')}
                    if ws.sock:
                        res = ws.sock.send(json.dumps(frame), websocket.ABNF.OPCODE_TEXT)
                        print("ali-send3->", res, json.dumps(frame))
                    else:
                        print("ali-send3-error->Connection is already closed", )
                    with wave.open(f'cache_data/{self.video_stream.window.shuziren["id"]}/audio/'+self.audio_id+'input2.wav', 'wb') as wf:
                        # 设置音频参数
                        n_channels = 1  # 单声道
                        sampwidth = 2  # 16 位音频，每个采样点 2 字节
                        wf.setnchannels(n_channels)
                        wf.setsampwidth(sampwidth)
                        wf.setframerate(16000)
                        wf.writeframes(self.data)
                    self.data = b''
            except Exception as e:
                print("ali-send-error->", e)
        # 单独开一个线程执行上面的run子方法
        thread.start_new_thread(run, ())

    def __connect(self):
        self.finalResults = ""
        self.done = False
        with self.lock:
            self.__frames.clear()
        self.__ws = websocket.WebSocketApp(self.__URL + '?token=' + _token, on_message=self.on_message)
        self.__ws.on_open = self.on_open
        self.__ws.on_close = self.on_close
        self.__ws.on_error = self.on_error
        self.__ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def add_frame(self, frame):
        with self.lock:
            self.__frames.append(frame)

    def send(self, buf):
        with self.lock:
            self.__frames.append(buf)

    def start(self):
        Thread(target=self.__connect, args=[]).start()
        data = {
            'header': self.__create_header('StartTranscription'),
            "payload": {
                "format": "pcm",
                "sample_rate": 16000,
                "enable_intermediate_result": True,
                "enable_punctuation_prediction": False,
                "enable_inverse_text_normalization": False,
                "enable_semantic_sentence_detection":False,
                "speech_noise_threshold": -0.5
            }
        }
        self.add_frame(data)

    def end(self):
        print("ali-end-->",self.__connected)
        # time.sleep(0.4)
        # if self.__connected:
        #     try:
        #         # for frame in self.__frames:
        #         #     self.__frames.popleft()
        #         #     if type(frame) == dict:
        #         #         self.__ws.send(json.dumps(frame))
        #         #         print("ali-end-send->", json.dumps(frame))
        #         #     elif type(frame) == bytes:
        #         #         self.__ws.send(frame, websocket.ABNF.OPCODE_BINARY)
        #         #         print("ali-end-send-byte>")
        #         #     time.sleep(0.04)
        #         self.__frames.clear()
        #         frame = {"header": self.__create_header('StopTranscription')}
        #         self.__ws.send(json.dumps(frame))
        #     except Exception as e:
        #         print("ali-end-error->",e)
        #         frame = {"header": self.__create_header('StopTranscription')}
        #         self.__ws.send(json.dumps(frame))
        self.__connected = False
        # with wave.open('cache_data/input2'+str(time.time())+'.wav', 'wb') as wf:
        #     # 设置音频参数
        #     n_channels = 1  # 单声道
        #     sampwidth = 2   # 16 位音频，每个采样点 2 字节
        #     wf.setnchannels(n_channels)
        #     wf.setsampwidth(sampwidth)
        #     wf.setframerate(16000)
        #     wf.writeframes(self.data)
        # self.data = b''
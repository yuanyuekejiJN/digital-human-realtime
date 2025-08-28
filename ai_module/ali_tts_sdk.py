import json
import os
import time
import threading
import uuid
from pathlib import Path

import nls
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

from scheduler.thread_manager import MyThread
from utils import config_util

__running = True
_token = ""


# 以下代码会根据上述TEXT文本反复进行语音合成
class AliTTS:
    def __get_history(self, voice_name, style, text):
        for data in self.__history_data:
            if data[0] == voice_name and data[1] == style and data[2] == text:
                return data[3]
        return None

    def __init__(self, tid, test_file, voice):
        self.__th = threading.Thread(target=self.__test_run)
        self.__id = tid
        self.__test_file = test_file
        self.__voice = voice
        self.__aformat = 'wav'  # 设置输出文件的格式

    def start(self, text):
        self.__text = text
        self.__f = open(self.__test_file, "wb")
        self.__th.start()
        self.__th.join()
        print(text)


    def test_on_metainfo(self, message, *args):
        print("on_metainfo message=>{}".format(message))

    def test_on_error(self, message, *args):
        print("on_error args=>{}".format(args))
        print("on_error:args=>{} message=>{}".format(args, message))

    def test_on_close(self, *args):
        print("on_close: args=>{}".format(args))
        try:
            self.__f.close()
        except Exception as e:
            print("close file failed since:", e)

    def test_on_data(self, data, *args):
        try:
            self.__f.write(data)
        except Exception as e:
            print("write data failed:", e)

    def test_on_completed(self, message, *args):
        print("on_completed:args=>{} message=>{}".format(args, message))

    def __test_run(self):
        global _token
        print("thread:{} start..".format(self.__id))
        tts = nls.NlsSpeechSynthesizer(url=config_util.key_ali_tts_url,
                                       token=_token,
                                       appkey=config_util.key_ali_tts_app_key,
                                       long_tts=True,
                                       on_metainfo=self.test_on_metainfo,
                                       on_data=self.test_on_data,
                                       on_completed=self.test_on_completed,
                                       on_error=self.test_on_error,
                                       on_close=self.test_on_close,
                                       callback_args=[self.__id])
        print("{}: session start".format(self.__id))
        print(self.__text)
        r = tts.start(self.__text, voice=self.__voice, aformat=Path(self.__test_file).suffix[1:])
        print("{}: tts done with result:{}".format(self.__id, r))


def text2wav(text, voice=None, filepath=None):
    if voice is None:
        voice = config_util.default_voice
    print("当前获取到的声音为：", voice,filepath)
    # if filepath is None:
    #     filepath = 'samples/sample-' + str(int(time.time() * 1000)) + '.mp3'
    t = AliTTS("thread-" + uuid.uuid4().hex, filepath, voice)
    t.start(text)
    return filepath


# 获取token信息
def __post_token():
    global _token
    __client = AcsClient(
        config_util.key_ali_tts_key_id,
        config_util.key_ali_tts_key_secret,
        "cn-shanghai"
    )

    __request = CommonRequest()
    __request.set_method('POST')
    __request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    __request.set_version('2019-02-28')
    __request.set_action_name('CreateToken')
    _token = json.loads(__client.do_action_with_exception(__request))['Token']['Id']


# 获取token任务（每隔12小时获取一次）
def __runnable():
    while __running:
        __post_token()
        time.sleep(60 * 60 * 12)


# 启动获取token的任务
def start():
    MyThread(target=__runnable).start()
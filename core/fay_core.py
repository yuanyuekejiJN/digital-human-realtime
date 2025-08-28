import concurrent.futures
import copy
import difflib
import hashlib
import imp
import math
import os
import queue
import random
import shutil
import threading
import time
import wave
import socket
import json
from pypinyin import lazy_pinyin

import cv2
import eyed3
from openpyxl import load_workbook
import logging

# 适应模型使用
import numpy as np
# import tensorflow as tf
import fay_booter
from ai_module import xf_ltp, nlp_langchao
from ai_module.ali_tts_sdk import text2wav
# from ai_module.ms_tts_sdk import Speech
from core import wsa_server, tts_voice, song_player
from core.interact import Interact
from core.tts_voice import EnumVoice
from scheduler.thread_manager import MyThread
from utils import util, storer, config_util
from core import qa_service

import pygame
from utils import config_util as cfg
from core.content_db import Content_Db
from datetime import datetime

from ai_module import nlp_rasa
from ai_module import nlp_chatgpt
from ai_module import nlp_gpt
from ai_module import nlp_yuan
# from ai_module import yolov8
from ai_module import nlp_VisualGLM
from ai_module import nlp_lingju
from ai_module import nlp_rwkv_api
from ai_module import nlp_ChatGLM2
from ai_module import nlp_dify

# 引入本地打包的wav2lip
from wav2lip.inference_torch import Wav2lip, getWav2lip
# 切割音频文件所用
from pydub import AudioSegment

import platform

if platform.system() == "Windows":
    import sys

    # sys.path.append("test/ovr_lipsync")
    # from test_olipsync import LipSyncGenerator

modules = {
    "nlp_yuan": nlp_yuan,
    "nlp_gpt": nlp_gpt,
    "nlp_langchao": nlp_langchao,
    "nlp_chatgpt": nlp_chatgpt,
    "nlp_rasa": nlp_rasa,
    "nlp_VisualGLM": nlp_VisualGLM,
    "nlp_lingju": nlp_lingju,
    "nlp_rwkv_api": nlp_rwkv_api,
    "nlp_chatglm2": nlp_ChatGLM2,
    "nlp_dify": nlp_dify

}

THINK_REMINDER = 15  # 需要提醒想一想的最小阈值
THINK_MP3_FILEPATH_LIST = ['mp3/think1.wav', 'mp3/think2.wav', 'mp3/think3.wav', 'mp3/think4.wav', 'mp3/think5.wav',
                           'mp3/think6.wav', 'mp3/think7.wav', 'mp3/think8.wav', 'mp3/think9.wav']
THINK_MP4_FILENAME_LIST = ['think1', 'think2', 'think3', 'think4', 'think5', 'think6', 'think7', 'think8',
                           'think9']  # 提醒音频文件位置

lock = threading.Lock()  # 创建一个锁对象


def determine_nlp_strategy(sendto, msg):
    text = ''
    textlist = []
    try:
        util.log(1, '自然语言处理...')
        tm = time.time()
        # cfg.load_config()
        if sendto == 2:
            text = nlp_chatgpt.question(msg)
        else:
            module_name = "nlp_" + cfg.key_chat_module
            print(module_name)
            selected_module = modules.get(module_name)
            if selected_module is None:
                raise RuntimeError('灵聚key、yuan key、gpt key都没有配置！')
            if cfg.key_chat_module == 'rasa':
                textlist = selected_module.question(msg)
                text = textlist[0]['text']
            else:
                text = selected_module.question(msg)
            util.log(1, '自然语言处理完成. 耗时: {} ms'.format(math.floor((time.time() - tm) * 1000)))
            if text == '哎呀，你这么说我也不懂，详细点呗' or text == '':
                util.log(1, '[!] 自然语言无语了！')
                text = '哎呀，你这么说我也不懂，详细点呗'
    except BaseException as e:
        print(e)
        util.log(1, '自然语言处理错误！')
        text = '哎呀，你这么说我也不懂，详细点呗'
    # print("fay core 117 text ----", text)
    # print("fay core 117 textlist ----", textlist)
    return text, textlist


# 文本消息处理
def send_for_answer(msg, sendto):
    contentdb = Content_Db()
    contentdb.add_content('member', 'send', msg)
    textlist = []
    text = None
    # 人设问答
    keyword = qa_service.question('Persona', msg)
    if keyword is not None:
        text = config_util.config["attribute"][keyword]

    # 全局问答
    if text is None:
        answer = qa_service.question('qa', msg)
        if answer is not None:
            text = answer
        else:
            text, textlist = determine_nlp_strategy(sendto, msg)

    contentdb.add_content('fay', 'send', text)
    wsa_server.get_web_instance().add_cmd({"panelReply": {"type": "fay", "content": text}})
    if len(textlist) > 1:
        i = 1
        while i < len(textlist):
            contentdb.add_content('fay', 'send', textlist[i]['text'])
            wsa_server.get_web_instance().add_cmd({"panelReply": {"type": "fay", "content": textlist[i]['text']}})
            i += 1
    return text


class FeiFei():
    def __init__(self):
        pygame.mixer.init()
        self.q_msg = ''
        self.a_msg = ''
        self.mood = 0.0  # 情绪值
        self.old_mood = 0.0
        self.connect = False
        self.item_index = 0
        self.deviceSocket = None
        self.deviceConnect = None

        # 启动音频输入输出设备的连接服务
        self.deviceSocketThread = MyThread(target=self.__accept_audio_device_output_connect)
        self.deviceSocketThread.start()

        self.X = np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)  # 适应模型变量矩阵
        # self.W = np.array([0.01577594,1.16119452,0.75828,0.207746,1.25017864,0.1044121,0.4294899,0.2770932]).reshape(-1,1) #适应模型变量矩阵
        self.W = np.array([0.0, 0.6, 0.1, 0.7, 0.3, 0.0, 0.0, 0.0]).reshape(-1, 1)  # 适应模型变量矩阵

        self.wsParam = None
        self.wss = None
        # self.sp = Speech()
        self.speaking = False
        self.last_interact_time = time.time()
        self.last_speak_data = ''
        self.interactive = []
        self.executor_core = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.executor_lip = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.sleep = False
        self.__running = True
        # self.sp.connect()  # 预连接
        self.last_quest_time = time.time()
        self.playing = False
        self.change_muting(cfg.config["attribute"]["open_huanxingchi"])
        self.video_stream = config_util.video_stream
        # threading.Thread(target=self.getWav2lip).start()
        self.wav2lip = getWav2lip()
        self.cache_mode = int(config_util.mp4_cache_mode)
        # self.timer = threading.Timer(60, self.change_muting, (True,))
        self.last_question = ""

    def getWav2lip(self):
        self.wav2lip = getWav2lip()
    def change_pkl(self):
        pass
        # face = self.video_stream.window.face
        # pkl_data_path = self.video_stream.window.pkl_data_path
        # self.wav2lip = Wav2lip(checkpoint_path='resource/checkpoints/yywav2lip.pth',
        #                        yolo_path='resource/yolov8n-face/yolov8n-face.pt',
        #                        facial_path='resource/wflw/hrnet18_256x256_p1/',
        #                        data_path='datas',
        #                        video_stream=config_util.video_stream,  # 传输video_stream
        #                        # face='resource/data/train.mp4',
        #                        face=face,
        #                        # pkl_data_path='resource/data/train.pkl',
        #                        pkl_data_path=pkl_data_path,
        #                        wav2lip_batch_size=config_util.batch_size)

    def __play_song(self):
        print("playsong asdasasdasdad+++++")
        self.playing = True
        song_player.play()
        self.playing = False
        wsa_server.get_web_instance().add_cmd({"panelMsg": ""})
        if not cfg.config["interact"]["playSound"]:  # 非展板播放
            content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': ""}}
            wsa_server.get_instance().add_cmd(content)

    # 修改是否为静音模式
    def change_muting(self, muting):
        print("修改是否为静音模式", muting, 111)
        config_util.config["interact"]["playSound"] = muting
        # config_util.save_config(config_util.config)

    # 检查是否命中指令或q&a
    def __get_answer(self, interleaver, text):

        # 把文字转成拼音
        # pinyin_list = lazy_pinyin(text)
        # if 'yuan' not in pinyin_list:
        #     return "请带关键词小元提问"
        if (
                cfg.config["attribute"]["open_huanxingchi"] and (
                cfg.config["attribute"]["huanxingchi_r"] in ''.join(lazy_pinyin(self.q_msg)))
        ):
            self.change_muting(False)
            # self.speak_time
            return "S_OPEN"

        else:
            pass
        if interleaver == "mic":
            # self.change_muting(cfg.config["attribute"]["open_huanxingchi"])
            # 指令
            keyword = qa_service.question('command', text)
            if keyword is not None:
                if keyword == "playSong":
                    MyThread(target=self.__play_song).start()
                    return "NO_ANSWER"
                elif keyword == "stop":
                    fay_booter.stop()
                    wsa_server.get_web_instance().add_cmd({"liveState": 0})
                    return "NO_ANSWER"
                elif keyword == "mute":
                    self.change_muting(True)
                    return "NO_ANSWER"
                elif keyword == "unmute":
                    self.change_muting(False)
                    return "S_OPEN"
                elif keyword == "changeVoice":
                    voice = tts_voice.get_voice_of(config_util.config["attribute"]["voice"])
                    for v in tts_voice.get_voice_list():
                        if v != voice:
                            config_util.config["attribute"]["voice"] = v.name
                            break
                    config_util.save_config(config_util.config)
                    return "NO_ANSWER"
            else:
                pass
        # if (
        #         (cfg.config["attribute"]["open_chufachi"] and (cfg.config["attribute"]["chufachi_r"]  in "".join(lazy_pinyin(self.q_msg))))
        # ):
        #     # self.change_muting(False)
        #     pass
        #     # return "NO_ANSWER"
        #     # self.change_muting(True)
        # else:
        #     # pass
        #     return "NO_ANSWER"
        # self.change_muting(False)
        # 人设问答
        keyword = qa_service.question('Persona', text)
        if keyword is not None:
            return config_util.config["attribute"][keyword]

        if config_util.q_a_list and len(config_util.q_a_list) > 0:
            for res in config_util.q_a_list:
                if text == res.get("question"):
                    va = res.get("answer")
                    if va:
                        return va
        # 全局问答
        return qa_service.question('qa', text)

    def __auto_speak(self, interact: Interact = None):
        if self.__running:
            # time.sleep(0.8)
            # print("-speaking->",self.speaking)
            # if self.speaking or self.sleep:
            #     continue
            try:
                # 如果有互动信息，就会走下面的逻辑
                # if self.video_stream.window.changes:
                #     self.change_pkl()
                #     self.video_stream.window.changes = False
                # if self.video_stream.window.changess:
                #     self.change_pkl()
                #     self.video_stream.window.changess = False
                if interact:
                    # interact: Interact = self.interactive.pop()
                    index = interact.interact_type
                    if index == 1:

                        # thread = threading.Thread(
                        #     target=lambda: (time.sleep(1), self.video_stream.window.pause_signal.emit(False)))
                        # thread.start()

                        self.q_msg = interact.data["msg"]
                        # 把文字转成拼音  检测到说你好  或者小元
                        # if cfg.config["attribute"]["open_chufachi"] and (cfg.config["attribute"]["chufachi"] not in lazy_pinyin(self.q_msg)):
                        #     continue
                        # self.video_stream.window.pause_signal.emit(True)
                        # time.sleep(2)
                        if "q_msg" in interact.data:
                            answer = interact.data["msg"]
                            self.a_msg = answer
                            self.q_msg = interact.data["q_msg"]
                            a_flag = str(self.q_msg)
                            # if a_flag != config_util.get_last_question():
                            #     return
                            # config_util.set_last_question2(a_flag)
                            self.__say(interact, a_flag)
                            # if cfg.config["attribute"]["open_huanxingchi"]:
                            #     self.change_muting(True)
                            contentdb = Content_Db()
                            contentdb.add_content('fay', 'speak', self.a_msg)
                            # print("fay core 339 msg:", self.a_msg)
                            wsa_server.get_web_instance().add_cmd(
                                {"panelReply": {"type": "fay", "content": self.a_msg}})
                            # wsa_server.get_web_instance().add_cmd(
                            #     {"panelReply": {"type": "fay", "content": "测试"}})
                            # if len(textlist) > 1:
                            #     i = 1
                            #     while i < len(textlist):
                            #         contentdb.add_content('fay', 'speak', textlist[i]['text'])
                            #         wsa_server.get_web_instance().add_cmd(
                            #             {"panelReply": {"type": "fay", "content": textlist[i]['text']}})
                            #         i += 1

                            wsa_server.get_web_instance().add_cmd({"panelMsg": self.a_msg})
                            if not cfg.config["interact"]["playSound"]:  # 非展板播放
                                content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': self.a_msg}}
                                wsa_server.get_instance().add_cmd(content)
                            return
                        else:
                            answer = self.__get_answer(interact.interleaver, self.q_msg)  # 确定是否命中指令或q&a

                        if not config_util.config["interact"]["playSound"]:  # 非展板播放
                            content = {'Topic': 'Unreal', 'Data': {'Key': 'question', 'Value': self.q_msg}}
                            wsa_server.get_instance().add_cmd(content)

                        if cfg.config["interact"]["playSound"]:  # 非展板播放
                            wsa_server.get_web_instance().add_cmd({"panelMsg": "静音指令正在执行，不互动"})
                            if not cfg.config["interact"]["playSound"]:  # 非展板播放
                                content = {'Topic': 'Unreal',
                                           'Data': {'Key': 'log', 'Value': "静音指令正在执行，不互动"}}
                                wsa_server.get_instance().add_cmd(content)
                            self.video_stream.wait_playing = False  # 清空等待状态
                            self.video_stream.window.update_listening(True)
                            return

                        if answer == 'NO_ANSWER':
                            return

                        if answer == "S_OPEN":
                            answer = "S_OPEN"
                        elif cfg.config["attribute"]["open_huanxingchi"]:
                            self.change_muting(True)
                            # pass
                            # self.timer.cancel()
                            # self.timer = threading.Timer(30, self.change_muting,(True,))
                            # self.timer.start()
                            # self.change_muting(True)
                        # self.timer.cancel()
                        # self.timer = threading.Timer(60, self.change_muting, (True,))
                        # self.timer.start()
                        contentdb = Content_Db()
                        contentdb.add_content('member', 'speak', self.q_msg)
                        wsa_server.get_web_instance().add_cmd({"panelReply": {"type": "member", "content": self.q_msg}})

                        # BRONCOS--QUESTION 标记
                        print("每次的提问问题：  ", self.q_msg)
                        # self.video_stream.window.update_listening_tingx(True)
                        self.video_stream.window.update_listening_dengdaix(True)
                        self.video_stream.window.update_listening_sikao(False)

                        self.video_stream.window.answer_signal.emit("")
                        self.video_stream.window.question_signal.emit(self.q_msg)  # 更新问题

                        text = ''
                        textlist = []
                        self.speaking = True
                        # if answer == "S_OPEN":
                        #     text = "我在有什么需要"

                        a_flag = self.q_msg
                        config_util.set_last_question(a_flag)
                        use_cache = config_util.get_cache(self.q_msg)
                        print("-->use_cache", use_cache)
                        if answer == "S_OPEN":
                            text = "我在有什么需要"
                        elif answer is not None and answer != 'NO_ANSWER' and use_cache:  # 语音内容没有命中指令,回复q&a内容
                            text = answer
                        else:
                            if use_cache:
                                cache_name = hashlib.md5(self.q_msg.encode('utf-8')).hexdigest()
                                # print("缓存文件名称：", cache_name)
                                path = f'cache_data/{self.video_stream.window.shuziren["id"]}/text_temp/' + cache_name
                                # print("缓存文件路径：", path)
                                if os.path.exists(path):
                                    with open(path, 'r') as f:
                                        text = f.read()
                                if len(text) <= 0:
                                    text, textlist = self.get_gpt_answer(a_flag)
                                    # print("调用了大模型进行回复text", text)
                                    # print("调用了大模型进行回复textlist", textlist)
                                    if len(text) > 0:
                                        # 缓存文字
                                        threading.Thread(target=self.cache_text_temp, args=(text, path)).start()
                                    return
                                else:
                                    buffer = ""
                                    for content in text:
                                        buffer += content
                                        if content in [".", "!", "?", "。", "！", "？",".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n", "！\\n\\n", "？\\n\\n","\\n\\n"]:
                                            print("--ans2>", buffer)
                                            threading.Thread(target=fay_booter.speak, args=(self.q_msg, buffer)).start()
                                            buffer = ""
                                    return

                            else:
                               text, textlist = self.get_gpt_answer(a_flag)
                               return

                        self.a_msg = text

                        # config_util.set_last_question(self.a_msg)
                        # 这是记录的
                        contentdb.add_content('fay', 'speak', self.a_msg)
                        wsa_server.get_web_instance().add_cmd({"panelReply": {"type": "fay", "content": self.a_msg}})
                        if len(textlist) > 1:
                            i = 1
                            while i < len(textlist):
                                contentdb.add_content('fay', 'speak', textlist[i]['text'])
                                wsa_server.get_web_instance().add_cmd(
                                    {"panelReply": {"type": "fay", "content": textlist[i]['text']}})
                                i += 1

                    wsa_server.get_web_instance().add_cmd({"panelMsg": self.a_msg})
                    if not cfg.config["interact"]["playSound"]:  # 非展板播放
                        content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': self.a_msg}}
                        wsa_server.get_instance().add_cmd(content)
                    self.last_speak_data = self.a_msg
                    # a_flag = str(time.time())
                    # MyThread(target=self.__say, args=['interact']).start()
                    self.__say(interact, a_flag)
            except BaseException as e:
                print(e)
            except ValueError as e:
                print(e)
                return

    def cache_text_temp(self, text,path):
        # 缓存文字
        if text is not None and text != '' and not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(text)

    def get_gpt_answer(self,a_flag):
        wsa_server.get_web_instance().add_cmd({"panelMsg": "思考中..."})
        if not cfg.config["interact"]["playSound"]:  # 非展板播放
            content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': "思考中..."}}
            wsa_server.get_instance().add_cmd(content)
        # 获取GPT的答案
        # self.video_stream.window.update_listening_sikao(True)
        threading.Thread(target=self.think, args=(a_flag,)).start()
        text, textlist = determine_nlp_strategy(1, self.q_msg)
        return text, textlist

    def think(self, a_flag):
        cur_voice = self.video_stream.window.cur_voice
        cache_path = self.video_stream.window.cache_path
        # if not mp4_cached and len(self.a_msg) > THINK_REMINDER:
        print("[info] 容我想一想提醒...")
        self.video_stream.xiangyixiang = True
        # if len(self.a_msg) > THINK_REMINDER:
        # 获取对应的句子
        think_text, think_random_num = self.video_stream.think_text_random()
        think_mp3_filepath = cache_path + '/' + THINK_MP3_FILEPATH_LIST[think_random_num]
        think_mp4_filepath = THINK_MP4_FILENAME_LIST[think_random_num]
        if not os.path.exists(think_mp3_filepath):
            text2wav(think_text, filepath=think_mp3_filepath, voice=cur_voice)
            # shutil.copy(think_mp3_filepath, think_mp3_filepath.replace('.mp3', '.wav'))
        think_mp4_cached, think_mp4_cache_path = self.check_mp4_cache(cache_path, think_mp4_filepath)
        # config_util.set_last_question(think_text)
        if not think_mp4_cached:
            # 这里需要videoStream里判断frame_list的内容，如果命中缓存且cache_mode为3时给定值为1，表示里面存储的是cap
            # 否则给定值0，表示里面存储的是帧

            print("视频已暂停wav2lip")
            frame_list = self.wav2lip.wav2lip(think_mp3_filepath,
                                              think_text, 0, a_flag=a_flag, is_think=False)
            # 存入缓存
            self.write_mp4_cache(think_mp4_cache_path.replace('mp4_temp', 'mp4'), frame_list)
        else:
            print("视频已暂停cache")
            frame_list = self.read_mp4_cache(think_mp4_cache_path)
            self.video_stream.offer(think_mp3_filepath, frame_list, think_text,
                                    1 if self.cache_mode == 3 else 0,segment_code=1, a_flag=a_flag, is_think=False)

    def speak(self, q_msg, text):
        # self.change_muting(False)
        interact: Interact = Interact("mic", 1, {"msg": text, "q_msg": q_msg, "user": "常见问题"})
        # self.interactive.append(interact)
        self.executor_core.submit(self.__auto_speak, interact)
        # self.__auto_speak(interact)
        # MyThread(target=self.__auto_speak).start()
        # MyThread(target=self.__update_mood, args=[interact.interact_type]).start()
        # MyThread(target=storer.storage_live_interact, args=[interact]).start()
        # index = interact.interact_type
        # if index == 1:
        #     self.q_msg = interact.data["msg"]
        #     # 把文字转成拼音  检测到说你好  或者小元
        #     # if cfg.config["attribute"]["open_chufachi"] and (cfg.config["attribute"]["chufachi"] not in lazy_pinyin(self.q_msg)):
        #     #     continue
        #     # self.video_stream.window.pause_signal.emit(True)
        #     # time.sleep(2)
        #     answer = self.__get_answer(interact.interleaver
        # self.a_msg = text
        # self.speaking = True
        #
        # cache_name = hashlib.md5(self.q_msg.encode('utf-8')).hexdigest()
        #
        # path = 'cache_data_one\\text_temp\\' + cache_name
        #
        # # 缓存文字
        # if self.a_msg is not None and self.a_msg != '' and not os.path.exists(path):
        #     with open(path, 'w') as f:
        #         f.write(self.a_msg)
        # # 这是记录的
        # # contentdb.add_content('fay', 'speak', self.a_msg)
        # wsa_server.get_web_instance().add_cmd({"panelReply": {"type": "fay", "content": self.a_msg}})
        #
        # wsa_server.get_web_instance().add_cmd({"panelMsg": self.a_msg})
        # if not cfg.config["interact"]["playSound"]:  # 非展板播放
        #     content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': self.a_msg}}
        #     wsa_server.get_instance().add_cmd(content)
        # self.last_speak_data = self.a_msg
        # MyThread(target=self.__say, args=['interact']).start()

    def on_interact(self, interact: Interact):
        # self.interactive.append(interact)
        # threading.Thread(target=self.__auto_speak, args=[interact]).start()
        threading.Thread(target=self.__auto_speak, args=[interact]).start()
        # self.__auto_speak(interact)
        # MyThread(target=self.__auto_speak).start()
        # MyThread(target=self.__update_mood, args=[interact.interact_type]).start()
        MyThread(target=storer.storage_live_interact, args=[interact]).start()

    # def on_interact_record(self, interact: Interact):
    #     # self.interactive.append(interact)
    #     threading.Thread(target=self.__auto_speak, args=[interact]).start()
    #     # self.executor_core.submit(self.__auto_speak, interact)
    #     # MyThread(target=storer.storage_live_interact, args=[interact]).start()
    #     # self.__auto_speak(interact)
    #     # MyThread(target=self.__auto_speak).start()
    #     # MyThread(target=self.__update_mood, args=[interact.interact_type]).start()
    #     MyThread(target=storer.storage_live_interact, args=[interact]).start()

    # 适应模型计算(用于学习真人的性格特质，开源版本暂不使用)
    def __fay(self, index):
        if 0 < index < 8:
            self.X[0][index] += 1
        # PRED = 1 /(1 + tf.exp(-tf.matmul(tf.constant(self.X,tf.float32), tf.constant(self.W,tf.float32))))
        PRED = np.sum(self.X.reshape(-1) * self.W.reshape(-1))
        if 0 < index < 8:
            print('***PRED:{0}***'.format(PRED))
            print(self.X.reshape(-1) * self.W.reshape(-1))
        return PRED

    # 发送情绪
    def __send_mood(self):
        while self.__running:
            time.sleep(3)
            if not self.sleep and not config_util.config["interact"][
                "playSound"] and wsa_server.get_instance().isConnect:
                content = {'Topic': 'Unreal', 'Data': {'Key': 'mood', 'Value': self.mood}}
                if not self.connect:
                    wsa_server.get_instance().add_cmd(content)
                    self.connect = True
                else:
                    if self.old_mood != self.mood:
                        wsa_server.get_instance().add_cmd(content)
                        self.old_mood = self.mood

            else:
                self.connect = False

    # 更新情绪
    def __update_mood(self, typeIndex):
        perception = config_util.config["interact"]["perception"]
        if typeIndex == 1:
            try:
                result = xf_ltp.get_sentiment(self.q_msg)
                chat_perception = perception["chat"]
                if result == 2:
                    self.mood = self.mood + (chat_perception / 200.0)
                elif result == 0:
                    self.mood = self.mood - (chat_perception / 100.0)
            except BaseException as e:
                print("[System] 情绪更新错误！")
                print(e)

        elif typeIndex == 2:
            self.mood = self.mood + (perception["join"] / 100.0)

        elif typeIndex == 3:
            self.mood = self.mood + (perception["gift"] / 100.0)

        elif typeIndex == 4:
            self.mood = self.mood + (perception["follow"] / 100.0)

        if self.mood >= 1:
            self.mood = 1
        if self.mood <= -1:
            self.mood = -1

    def __get_mood_voice(self):
        voice = tts_voice.get_voice_of(config_util.config["attribute"]["voice"])
        if voice is None:
            voice = EnumVoice.XIAO_XIAO
        styleList = voice.value["styleList"]
        sayType = styleList["calm"]
        if -1 <= self.mood < -0.5:
            sayType = styleList["angry"]
        if -0.5 <= self.mood < -0.1:
            sayType = styleList["lyrical"]
        if -0.1 <= self.mood < 0.1:
            sayType = styleList["calm"]
        if 0.1 <= self.mood < 0.5:
            sayType = styleList["assistant"]
        if 0.5 <= self.mood <= 1:
            sayType = styleList["cheerful"]
        return sayType

    # 合成声音
    def __say(self, styleType, a_flag):
        global think_text, think_mp3_filepath, think_mp4_filepath
        if a_flag != config_util.get_last_question():
            return
        try:
            if len(self.a_msg) < 1:
                self.speaking = False
            else:
                util.printInfo(1, '菲菲', '({}) {}'.format(self.__get_mood_voice(), self.a_msg))
                # if not config_util.config["interact"]["playSound"]:  # 非展板播放
                content = {'Topic': 'Unreal', 'Data': {'Key': 'text', 'Value': self.a_msg}}
                wsa_server.get_instance().add_cmd(content)
                MyThread(target=storer.storage_live_interact,
                         args=[Interact('Fay', 0, {'user': 'Fay', 'msg': self.a_msg})]).start()

                util.log(1, '合成音频...')
                '''
                文件说明：
                    目录位于cache_data下
                    · mp4           所有固定缓存mp4文件存放的位置（事先放置）
                    · mp4_temp      运行中产生的mp4缓存文件位置
                    · wav           所有固定缓存wav文件存放的位置（事先放置）
                    · wav_temp      运行中产生的wav缓存文件位置
                代码逻辑：
                    - 先根据问题查找答案缓存（若无使用gpt生成）
                    - 根据回答查找是否有音频缓存、视频缓存
                    1. 将答案编码MD5
                    2. 查看是否有wav文件与之匹配（分别位于固定、临时位置）
                    3. 若有，则跳过下面部分逻辑，使用拷贝的方式放置在固定的文件夹下
                    4. 若无，继续向下执行
                注意：wav2lip只能接受wav文件，而音频播放器只能接受mp3文件，只改名即可使用。
                '''
                result = None  # 为None则表示未命中缓存
                mp4_cached = False
                cache_path = self.video_stream.window.cache_path
                # if self.video_stream.window.change_cache:
                #     cache_path = self.video_stream.window.cache_path
                #     self.video_stream.window.change_cache = False
                cache_name = hashlib.md5(self.a_msg.encode('utf-8')).hexdigest()
                # new_path = 'samples\\sample-' + str(int(time.time() * 1000))
                cur_voice = self.video_stream.window.cur_voice

                # 将缓存拷贝到samples文件夹中
                # def copy2samples(old_path, isAudio):
                #     nonlocal result, mp4_cached
                #     if isAudio:
                #         print("发现MP3缓存，复制中..." + old_path)
                #         shutil.copy(old_path, new_path + '.mp3')
                #         print("已命中MP3缓存")
                #         result = new_path + '.mp3'
                #     else:
                #         mp4_cached = True

                # 将mp3文件缓存到mp3_temp中
                # def save2cache(path):
                #     shutil.copy(path, cache_path + '\\mp3_temp\\' + cache_name + '.mp3')

                # 查看是否命中音频缓存
                # if os.path.exists(temp_file_name := cache_path + '\\mp3\\' + cache_name + '.mp3'):
                #     copy2samples(temp_file_name, True)
                # elif os.path.exists(temp_file_name := cache_path + '\\mp3_temp\\' + cache_name + '.mp3'):
                #     copy2samples(temp_file_name, True)

                # 查看是否存在MP4缓存文件
                mp4_cached, mp4_cache_path = self.check_mp4_cache(cache_path, cache_name)
                tm = time.time()
                # real_a_flag = self.a_msg + str(tm)
                # print("real_a_flag:", real_a_flag)
                # a_flag = hashlib.md5((real_a_flag).encode('utf-8')).hexdigest()
                # config_util.set_last_question(a_flag)
                # 想一想功能
                # if not mp4_cached and len(self.a_msg) > THINK_REMINDER:
                #     threading.Thread(target=self.think, args=(a_flag,)).start()
                #     print("[info] 容我想一想提醒...")
                #     self.video_stream.xiangyixiang = True
                #     if len(self.a_msg) > THINK_REMINDER:
                #         # 获取对应的句子
                #         think_text, think_random_num = self.video_stream.think_text_random()
                #         think_mp3_filepath = cache_path + '/' + THINK_MP3_FILEPATH_LIST[think_random_num]
                #         think_mp4_filepath = THINK_MP4_FILENAME_LIST[think_random_num]
                #     if not os.path.exists(think_mp3_filepath):
                #         text2wav(think_text, filepath=think_mp3_filepath, voice=cur_voice)
                #         shutil.copy(think_mp3_filepath, think_mp3_filepath.replace('.mp3', '.wav'))
                #     think_mp4_cached, think_mp4_cache_path = self.check_mp4_cache(cache_path, think_mp4_filepath)
                #     # config_util.set_last_question(think_text)
                #     if not think_mp4_cached:
                #         # 这里需要videoStream里判断frame_list的内容，如果命中缓存且cache_mode为3时给定值为1，表示里面存储的是cap
                #         # 否则给定值0，表示里面存储的是帧
                #
                #         print("视频已暂停wav2lip")
                #         frame_list = self.wav2lip.wav2lip(think_mp3_filepath.replace('.mp3', '.wav'),
                #                                           think_text, 0,a_flag=a_flag,is_think=False)
                #         # 存入缓存
                #         self.write_mp4_cache(think_mp4_cache_path.replace('mp4_temp', 'mp4'), frame_list)
                #     else:
                #         print("视频已暂停cache")
                #         frame_list = self.read_mp4_cache(think_mp4_cache_path)
                #         self.video_stream.offer(think_mp3_filepath, frame_list, think_text,
                #                                 1 if self.cache_mode == 3 else 0,a_flag=a_flag,is_think=False)

                # 文字也推送出去，为了ue5
                # 未命中时会执行下述逻辑
                if not os.path.exists(result := cache_path + '\\mp3_temp\\' + cache_name + '.wav'):
                    if config_util.tts_mode == 'ali':
                        result = text2wav(self.a_msg, voice=cur_voice,filepath=result)
                    # else:
                    #     # TODO: 这里暂未测试
                    #     result = self.sp.to_sample(self.a_msg, self.__get_mood_voice())

                # if result is not None:
                #     save2cache(result)
                #     wav_result = result.replace('.mp3', '.wav')
                #     if not os.path.exists(wav_result):
                #         shutil.copy(result, wav_result)

                util.log(1, '合成音频完成. 耗时: {} ms 文件:{}'.format(math.floor((time.time() - tm) * 1000),
                                                                       result))

                # MyThread(target=self.__send_or_play_audio,
                #          args=[wav_result, styleType, mp4_cached, mp4_cache_path]).start()
                # self.executor_lip.submit(self.__send_or_play_audio, wav_result, styleType, mp4_cached, mp4_cache_path, a_flag)
                self.__send_or_play_audio(result, styleType, mp4_cached, mp4_cache_path, a_flag)
                # self.video_stream.window.update_listening_dengdaix(True)
                # self.video_stream.window.update_listening_sikao(True)
                # self.video_stream.window.update_listening_shuox(False)
                return result

        except BaseException as e:
            print(e)
        self.speaking = False
        return None

    def __play_sound(self, file_url):
        util.log(1, '播放音频...' + file_url)
        util.log(1, '问答处理总时长：{} ms'.format(math.floor((time.time() - self.last_quest_time) * 1000)))
        pygame.mixer.music.load(file_url)
        pygame.mixer.music.play()

    def __send_or_play_audio(self, file_url, say_type, cached, cache_path, a_flag):
        try:
            try:
                logging.getLogger('eyed3').setLevel(logging.ERROR)
                # audio_length = eyed3.load(file_url).info.time_secs  # mp3音频长度
            except Exception as e:
                audio_length = 3

            # with wave.open(file_url, 'rb') as wav_file: #wav音频长度
            #     audio_length = wav_file.getnframes() / float(wav_file.getframerate())
            #     print(audio_length)
            # if audio_length <= config_util.config["interact"]["maxInteractTime"] or say_type == "script":
            # if config_util.config["interact"]["playSound"]:  # 展板播放
            #     print("不应该执行到这里！！！")
            #     self.__play_sound(file_url)
            # else:  # 发送音频给ue和socket
            # 这里需要videoStream里判断frame_list的内容，如果命中缓存且cache_mode为3时给定值为1，表示里面存储的是cap
            # 否则给定值0，表示里面存储的是帧
            mode = 1 if cached and self.cache_mode == 3 else 0

            '''
                关于wav2lip，需要传输音频文件地址，文字，模式
                    需要在wav2lip内部直接传输给video_stream

                关于缓存，依旧在这里处理、读取

            '''
            # 生成视频
            frame_list = []
            if not cached:

                frame_list = self.wav2lip.wav2lip(file_url,  # 音频文件地址
                                                  self.a_msg,  # 文字
                                                  mode, a_flag)  # 模式

                self.video_stream.wait_playing = False  # 清空等待状态
                self.video_stream.bofang = True
            else:
                self.video_stream.huancun = True
                print(f"[info] 命中缓存，正在读取缓存文件，地址：{cache_path}")
                frame_list = self.read_mp4_cache(cache_path)
                # print("在这里是播放缓存的！！！！！！！！！！！！！！！！！")
                self.video_stream.offer(file_url, frame_list, self.a_msg, mode, segment_code=1, a_flag=a_flag)
                self.video_stream.wait_playing = False  # 清空等待状态
                print("[info] 读取缓存文件成功")

            # 传入的参数为
            # print(f"传入的参数为：{file_url.replace('.wav', '.mp3')} , {len(frame_list)} , {self.a_msg}")
            print(f"[info] 已将新回答加入到播放列表中")

            if not cached:
                # 保存mp4缓存文件
                print(f"[info] 没有命中缓存，正在缓存mp4文件，正在生成缓存文件：{cache_path}")
                threading.Thread(target=self.write_mp4_cache,args=(cache_path, frame_list)).start()
                # self.write_mp4_cache(cache_path, frame_list)

            self.speaking = False
            print("推送视频线程结束")
        except Exception as e:
            print(e)

    # 将视频帧序列化为视频文件
    # EX: 这里固定了缓存视频的数量为9个，后续可以优化
    def serializing_video_frames(self, frames_list, filepath):
        if frames_list is None:
            return
        g = time.time()

        # 创建目录
        os.makedirs(filepath, exist_ok=True)

        # print(self.wav2lip.get_video_info())
        out_1 = cv2.VideoWriter(f"{filepath}\\1.avi", *self.wav2lip.get_video_info())
        out_2 = cv2.VideoWriter(f"{filepath}\\2.avi", *self.wav2lip.get_video_info())
        out_3 = cv2.VideoWriter(f"{filepath}\\3.avi", *self.wav2lip.get_video_info())
        out_4 = cv2.VideoWriter(f"{filepath}\\4.avi", *self.wav2lip.get_video_info())
        out_5 = cv2.VideoWriter(f"{filepath}\\5.avi", *self.wav2lip.get_video_info())
        out_6 = cv2.VideoWriter(f"{filepath}\\6.avi", *self.wav2lip.get_video_info())
        out_7 = cv2.VideoWriter(f"{filepath}\\7.avi", *self.wav2lip.get_video_info())
        out_8 = cv2.VideoWriter(f"{filepath}\\8.avi", *self.wav2lip.get_video_info())
        out_9 = cv2.VideoWriter(f"{filepath}\\9.avi", *self.wav2lip.get_video_info())

        half_length = len(frames_list) // 9
        frames_1 = frames_list[:half_length]
        frames_2 = frames_list[half_length: half_length * 2]
        frames_3 = frames_list[2 * half_length:3 * half_length]
        frames_4 = frames_list[3 * half_length:4 * half_length]
        frames_5 = frames_list[4 * half_length:5 * half_length]
        frames_6 = frames_list[5 * half_length:6 * half_length]
        frames_7 = frames_list[6 * half_length:7 * half_length]
        frames_8 = frames_list[7 * half_length:8 * half_length]
        frames_9 = frames_list[8 * half_length:]
        # frames_list.clear()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.process_frames, out_1, frames_1)
            executor.submit(self.process_frames, out_2, frames_2)
            executor.submit(self.process_frames, out_3, frames_3)
            executor.submit(self.process_frames, out_4, frames_4)
            executor.submit(self.process_frames, out_5, frames_5)
            executor.submit(self.process_frames, out_6, frames_6)
            executor.submit(self.process_frames, out_7, frames_7)
            executor.submit(self.process_frames, out_8, frames_8)
            executor.submit(self.process_frames, out_9, frames_9)
        # 等待所有线程完成
        while out_1.isOpened() or out_2.isOpened() or out_3.isOpened() or out_4.isOpened() or out_5.isOpened() or out_6.isOpened() or out_7.isOpened() or out_8.isOpened() or out_9.isOpened():
            time.sleep(0.1)
        k = time.time()
        print(f'[info] 生成缓存的时间为：{k - g}')

    def process_frames(self, file_out, frames_list):
        frame_queue = queue.Queue()
        for f in frames_list:
            frame_queue.put(f)

        self.__frames_write_local(file_out, frame_queue)

    # 根据缓存模式，检测cache是否存在
    # 返回一个元组(a, b)，a为是否有缓存，b为缓存的路径
    # 若无缓存，b为默认缓存路径
    def check_mp4_cache(self, cache_path, cache_name):
        suffix = ''
        if self.cache_mode == 1:
            suffix = '.bin'
        elif self.cache_mode == 2:
            suffix = '.npz'
        elif self.cache_mode == 3:
            suffix = ''

        cache_name = cache_name + suffix
        default_path = cache_path + '\\mp4_temp\\' + cache_name
        filepaths = [cache_path + '\\mp4\\' + cache_name, cache_path + '\\mp4_temp\\' + cache_name]
        for fp in filepaths:
            if os.path.exists(fp):
                return True, fp
        return False, default_path

    # 使用前需要使用 check_mp4_cache 获取文件名
    def write_mp4_cache(self, filename, content):
        if self.cache_mode == 1:
            # numpy普通模式
            with open(filename, 'wb') as f:
                np.save(f, content)
        elif self.cache_mode == 2:
            # numpy压缩模式
            with open(filename, 'wb') as f:
                np.savez_compressed(f, content)
        elif self.cache_mode == 3:
            try:
                self.serializing_video_frames(content, filename)
            except Exception as e:
                # TODO: 该部分待测试
                print(f"cv2写入视频失败，失败原因：{e}", e)
                try:
                    # 删除文件夹及其所有内容，防止下一次读取到
                    shutil.rmtree(filename)
                except OSError as e:
                    print(f"Error: {e.strerror}")

    # 使用前需要使用 check_mp4_cache 获取文件名
    def read_mp4_cache(self, filename):
        frame_list = None
        if self.cache_mode == 1:
            # numpy模式
            with open(filename, 'rb') as f:
                frame_list = np.load(f)
        elif self.cache_mode == 2:
            # numpy压缩模式
            with open(filename, 'rb') as f:
                frame_list = np.load(f)['arr_0']
        elif self.cache_mode == 3:
            # 检测文件夹下有多少个文件
            frame_list = []
            i = 1
            while os.path.exists(temp_path := f'{filename}\\{i}.avi'):
                frame_list.append(cv2.VideoCapture(temp_path))
                i += 1
        return frame_list

    def __frames_write_local(self, file_out, frame_queue):
        # 帧写入本地
        while True:
            # 从队列中获取帧
            frame = frame_queue.get()
            if frame is None:
                print('Error: frame is None')
                file_out.release()
                break
            file_out.write(frame)
            # 标记视频处理完成
            with lock:
                # 通知队列任务完成
                frame_queue.task_done()

                if frame_queue.unfinished_tasks == 0:
                    file_out.release()
                    break

    def __device_socket_keep_alive(self):
        while True:
            if self.deviceConnect is not None:
                try:
                    self.deviceConnect.send(b'\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8')  # 发送心跳包
                except Exception as serr:
                    util.log(1, "远程音频输入输出设备已经断开：{}".format(serr))
                    self.deviceConnect = None
            time.sleep(1)

    def __accept_audio_device_output_connect(self):
        self.deviceSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.deviceSocket.bind(("0.0.0.0", 10001))
        self.deviceSocket.listen(1)
        addr = None
        try:
            while True:
                self.deviceConnect, addr = self.deviceSocket.accept()  # 接受TCP连接，并返回新的套接字与IP地址
                MyThread(target=self.__device_socket_keep_alive).start()  # 开启心跳包检测
                util.log(1, "远程音频输入输出设备连接上：{}".format(addr))
                while self.deviceConnect:  # 只允许一个设备连接
                    time.sleep(1)
        except Exception as err:
            pass

    def set_sleep(self, sleep):
        self.sleep = sleep

    def start(self):
        pass
        # 每隔三秒向UE5发送一次情绪
        # MyThread(target=self.__send_mood).start()
        # 每隔0.8秒查看是否有互动信息
        # MyThread(target=self.__auto_speak).start()

    def stop(self):
        self.__running = False
        song_player.stop()
        self.speaking = False
        self.playing = False
        # self.sp.close()
        wsa_server.get_web_instance().add_cmd({"panelMsg": ""})
        if not cfg.config["interact"]["playSound"]:  # 非展板播放
            content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': ""}}
            wsa_server.get_instance().add_cmd(content)
        if self.deviceConnect is not None:
            self.deviceConnect.close()
            self.deviceConnect = None
        if self.deviceSocket is not None:
            self.deviceSocket.close()

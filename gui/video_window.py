import math
import os
import random
import sys
import tkinter
from collections import deque

import pickle

import markdown
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QByteArray, QMargins
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QApplication, QLabel, QGridLayout, QSizePolicy
import cv2
import threading
import time

from PyQt5.uic.properties import QtGui
from pydub import AudioSegment

import fay_booter
from core.question_db import Question_Db, question_db
from core.shuziren_db import shuziren_db
from core.soundPlayer import SoundPlayer
from gui.flowlaout import FlowLayout
from utils import config_util
from utils.util import printInfo
from wav2lip import inference_torch

DEFAULT_RESOURCE_LENGTH = 1
sys.setrecursionlimit(10000)


class VideoStream(QThread):
    def __init__(self, window):
        super().__init__()
        # 这是我自己的播放器  新的库
        self.is_think = None
        self.duwei_player = SoundPlayer()
        self.running = True  # 播放开关
        # self.window.default = f'resource/data/default/defaultPart0{x}.mp4'
        # self.window.default1 = f'resource/data/default/defaultPart0{x}.mp3'
        self.window = window
        self.answer_text = None
        default = self.window.default
        default1 = self.window.default1

        self.THINK_TEXT_LIST = [
            "嗯，容我想一想",
            # "问题有些复杂,我得好好的想一想",
            # "稍等片刻，我马上就有答案了",
            # "让我想一想，马上告诉你",
            # "请给我一点时间，我正在思考",
            # " 别急，我正在仔细考虑这个问题",
            # "稍安勿躁，我马上就能想出办法",
            # "请稍候，我正在脑海中搜索答案",
            # "稍等一下，我正在整理思绪"
        ]

        # 添加文件
        self.default_resource = []
        # for i in range(1, DEFAULT_RESOURCE_LENGTH + 1):
        #     # x = str(i).zfill(2)
        #     # cap = cv2.VideoCapture(f'resource/data/default/defaultPart0{x}.mp4')
        #     cap = cv2.VideoCapture(default)
        #     # audio = AudioSegment.from_file(f'resource/data/default/defaultPart0{x}.mp3', "mp3")
        #     # audio = AudioSegment.from_file(default1, "mp3")
        #     temp_frame_list = []
        #     while cap.isOpened():
        #         ret, frame = cap.read()
        #         if ret:
        #             # opencv resize
        #             frame = cv2.resize(frame, (1440, 3840))
        #             temp_frame_list.append(frame)
        #         else:
        #             break
        #
        #     self.default_resource.append((
        #         cap.get(cv2.CAP_PROP_FPS),
        #         temp_frame_list,
        #        None
        #     ))  # 默认资源文件 (缓存的依次为：rate、cap、mp3_path)
        #
        # # 倒放静止视频
        # for i in range(1, DEFAULT_RESOURCE_LENGTH + 1):
        #     prev = DEFAULT_RESOURCE_LENGTH - i
        #     self.default_resource.append((self.default_resource[prev][0],
        #                                   self.default_resource[prev][1][::-1],
        #                                   self.default_resource[prev][2]))

        self.playlist = deque()  # 播放列表
        self.textlist = deque()  # 字幕列表

        self.window = window

        self.playing = False

        self.wait_playing = False

        self.huancun = False

        self.huancun1 = False

        self.xiangyixiang = False

        self.xiangyixiang1 = False

        self.bofang = False

        self.bofang1 = False

        self.zhanshitext = ""

    def think_text_random(self):
        # num = random.randint(0, 8)
        text = self.THINK_TEXT_LIST[0]
        return text, 0

    # 播放默认视频
    def default_video(self):

        # if not self.wait_playing:
        #     self.window.update_listening(True)
        if self.playing:
            self.window.ui.label_7.setHidden(False)
        self.playing = False
        if self.window.change_jingyin:
            print("这里改变了静音", self.window.change_jingyin)
            self.window.change_jingyin = False
            self.default_resource.clear()
            self.default_resource = []
            for i in range(1, DEFAULT_RESOURCE_LENGTH + 1):
                # x = str(i).zfill(2)
                # cap = cv2.VideoCapture(f'resource/data/default/defaultPart0{x}.mp4')
                cap = cv2.VideoCapture(self.window.default)
                # audio = AudioSegment.from_file(f'resource/data/default/defaultPart0{x}.mp3', "mp3")
                # audio = AudioSegment.from_file(self.window.default1, "mp3")
                temp_frame_list = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        if height != config_util.video_height:
                            frame = cv2.resize(frame, (config_util.video_width, config_util.video_height))
                        temp_frame_list.append(frame)
                    else:
                        break

                self.default_resource.append((
                    cap.get(cv2.CAP_PROP_FPS),
                    temp_frame_list,
                    None
                ))  # 默认资源文件 (缓存的依次为：rate、cap、mp3_path)

            # 倒放静止视频
            for i in range(1, DEFAULT_RESOURCE_LENGTH + 1):
                prev = DEFAULT_RESOURCE_LENGTH - i
                self.default_resource.append((self.default_resource[prev][0],
                                              self.default_resource[prev][1][::-1],
                                              self.default_resource[prev][2]))
        if self.window.change_jingyins:
            self.window.change_jingyins = False
            self.default_resource.clear()
            self.default_resource = []
            default = self.window.default
            default1 = self.window.default1
            for i in range(1, DEFAULT_RESOURCE_LENGTH + 1):
                # x = str(i).zfill(2)
                # cap = cv2.VideoCapture(f'resource/data/default/defaultPart0{x}.mp4')
                cap = cv2.VideoCapture(default)
                # audio = AudioSegment.from_file(f'resource/data/default/defaultPart0{x}.mp3', "mp3")
                # audio = AudioSegment.from_file(default1, "mp3")
                temp_frame_list = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        if (height != config_util.video_height):
                            frame = cv2.resize(frame, (config_util.video_width, config_util.video_height))
                        temp_frame_list.append(frame)
                    else:
                        break

                self.default_resource.append((
                    cap.get(cv2.CAP_PROP_FPS),
                    temp_frame_list,
                    None
                ))  # 默认资源文件 (缓存的依次为：rate、cap、mp3_path)

            # 倒放静止视频
            for i in range(1, DEFAULT_RESOURCE_LENGTH + 1):
                prev = DEFAULT_RESOURCE_LENGTH - i
                self.default_resource.append((self.default_resource[prev][0],
                                              self.default_resource[prev][1][::-1],
                                              self.default_resource[prev][2]))

    # 运行检测
    def run(self):
        default_cur = 0
        self.mode = 0
        segment_code = 0
        while self.running:
            if self.bofang1:
                self.window.update_listening_shuox(True)
                self.window.update_listening_dengdaix(False)
                self.window.setplaying = True
                self.bofang1 = False
                if self.xiangyixiang1:
                    self.window.update_listening_shuox(True)
                    self.window.update_listening_dengdaix(True)
                    self.window.update_listening_sikao(False)
                    self.xiangyixiang1 = False

            self.answer_text = None
            a_flag = None
            self.is_think = False
            total_time = None
            if len(self.playlist) > 0:

                default_cur = 0
                self.playing = True
                self.bofang = True
                mp3_path, frame_list, self.answer_text, mode, total_frame_size, segment_code, a_flag, self.is_think = self.playlist.popleft()
                # print("vedio windows 223 answer text: ", self.answer_text)
                self.mode = mode
                # 计算帧率
                audio = AudioSegment.from_file(mp3_path)
                total_time = audio.duration_seconds  # 获取时长，单位为秒
                rate = total_frame_size / total_time  # 总帧数 / 音频总时间    按照音频时间播放
                self.textlist.append((self.answer_text,total_time))
                # 判断缓存模式
                if self.mode == 1:
                    rate = sum(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in frame_list) / total_time
            else:
                self.mode = 4  # 以加载帧后的方式加载静止视频，且填充末三位为1，标志位静止视频
                segment_code = 0
                # height, width = self.default_resource[0][1][0].shape[:2]
                # print(f"----->2Height: {height}, Width: {width}")
                self.default_video()
                rate, frame_list, mp3_path = self.default_resource[default_cur]
                # height, width = frame_list[0].shape[:2]
                # print(f"----->1Height: {height}, Width: {width}")
                default_cur = (default_cur + 1) % (DEFAULT_RESOURCE_LENGTH * 2)
                a_flag = mp3_path
            # print("-->segment_code-->",segment_code,"<-->",mp3_path,"<-->")
            if segment_code == 1:
                if mp3_path:
                    # 不管是缓存还是新的 声音 都用这个播放
                    threading.Thread(target=self.music_single, args=[mp3_path,a_flag]).start()
            if segment_code == 1 and total_time is not None:
                pass
                # self.thread_answer = threading.Thread(target=self.answer_text_single, args=[self.answer_text, total_time,a_flag])
                # self.thread_answer.start()
            if self.huancun:
                self.window.update_listening_sikao(True)
                self.window.update_listening_dengdaix(True)
                self.window.update_listening_shuox(False)
                self.huancun = False
                self.huancun1 = True
            if self.xiangyixiang:
                self.window.update_listening_sikao(True)
                self.window.update_listening_dengdaix(True)
                self.window.update_listening_shuox(False)
                self.xiangyixiang = False
                self.xiangyixiang1 = True
            if self.bofang:
                self.window.update_listening_sikao(True)
                self.window.update_listening_shuox(False)
                self.window.update_listening_dengdaix(True)
                self.window.setplaying = False
                self.bofang = False
                self.bofang1 = True
            # print(segment_code, mode)
            # self.thread_video = threading.Thread(target=self.video_single, args=[frame_list, rate, self.mode, segment_code])
            # self.thread_video.start()
            # height, width = frame_list[0].shape[:2]
            # print(f"----->Height: {height}, Width: {width}")
            self.video_single(frame_list, rate, self.mode, segment_code,a_flag)
            if self.huancun1:
                self.window.update_listening_shuox(True)
                self.window.update_listening_sikao(True)
                self.window.update_listening_dengdaix(False)
                self.huancun1 = False

    def start_txt(self, a_flag):
        self.thread_answer = threading.Thread(target=self.answer_text_single,
                                              args=[a_flag])
        self.thread_answer.start()
    def music_single(self, resource,a_flag):
        print("----------->play_music1-",a_flag,resource)
        self.duwei_player.load_sound(resource,a_flag)
        # self.duwei_player.play_sound(a_flag)
        # print("----------->play_music-",a_flag)

    # 清除音频  得
    def stop_music(self,a_flag):
        print("----------->stop_music1-",a_flag)
        res = self.duwei_player.stop_sound(a_flag)
        self.textlist.clear()
        if res is True:
            print("----------->stop_music-",a_flag)
            # if self.a_flag != config_util.get_last_question() and not self.is_think:
            #     print("----------->stop_music没有执行")
            #     return
            self.window.update_listening_shuox(True)
            # self.window.update_listening_tingx(True)
            self.window.update_listening_sikao(True)
            self.window.update_listening_dengdaix(False)
            # self.window.setplaying = True
            # self.bofang1 = False
            # self.playlist.clear()



    # 先播放一秒，如果当前时间对不上，视频就等一下，等音频跟上再继续播放
    # frame_list    帧列表
    # rate          帧率
    # mode          模式说明
    #        mode(低位到高位)：
    #           第1~2位代表存储模式
    #           第3位代表是否为默认视频
    #           第4位代表
    #
    def video_single(self, frame_list, rate, mode, segment_code,a_flag):
        # printInfo(1,"ui",'mode'+str(mode))
        # 4是 默认的
        # time.sleep(0.01)
        # return
        startTime = time.time()
        if mode & 3 == 0:
            frame_deque = deque()
            frame_deque.append(frame_list)
            while frame_deque:
                startTime = time.time()
                frame_list = frame_deque.popleft()
                n, i = len(frame_list), 0
                while i < n:
                    if mode != 4:
                        if a_flag != config_util.get_last_question() and not self.is_think:
                            threading.Thread(target=self.stop_music,args=(a_flag,)).start()
                            return
                    # height, width = frame_list[i].shape[:2]
                    # print(f"--->2Height: {height}, Width: {width}")
                    try:
                        config_util.frame.append(frame_list[i])
                        # byte_array = pickle.dumps(frame_list[i])
                    except Exception as e:
                        print("pickle.dumps(frame_list[i])-->", e)
                        return
                    except BaseException as e:
                        print("pickle.dumps(frame_list[i])-->", e)
                        return
                    # print(f'-frame_list--{i}->', len(byte_array))
                    self.window.video_signal.emit(QByteArray())
                    cv2.waitKey(1)  # 等待1毫秒 （1秒=1000毫秒）
                    if mode & 4 != 0:
                        if self.window.change_jingyin:
                            return
                        if len(self.playlist) > 0:
                            self.window.update_listening_sikao(True)
                            self.window.update_listening_dengdaix(True)
                            self.window.update_listening_shuox(False)
                            # 如果播放静态视频时检测到有新视频加入，则立即播放新视频
                            return
                    # TODO: 这里逻辑不太对，但是思路没问题，至于问题是否得到解决有待测试
                    elif segment_code >= 1 and len(self.playlist) > 0 and self.playlist[0][5] >= 1:
                        # pass
                        if self.playlist[0][5] == 1:
                            audio = AudioSegment.from_file(self.playlist[0][0])
                            total_time = audio.duration_seconds  # 获取时长，单位为秒
                            self.textlist.append((self.playlist[0][2], total_time))
                            self.duwei_player.load_sound(self.playlist[0][0], a_flag)

                        next_frame_list = self.playlist.popleft()[1]
                        frame_deque.append(next_frame_list)
                        # n += len(next_frame_list)
                    sleepTime = i / rate - time.time() + startTime
                    if sleepTime > 0:  # 播放时间快了就等一下
                        # print("睡眠：", sleepTime)
                        time.sleep(sleepTime)
                    i += 1
                # frame_list.clear()
        elif mode & 3 == 1:
            # 当缓存模式为3时，frame_list里存储的是多个cap
            if not (rate >0):
                return
            e = cur = 0
            for cap in frame_list:
                # print("cap",cap)
                while cap.isOpened():
                    if a_flag != config_util.get_last_question() and not self.is_think:
                        threading.Thread(target=self.stop_music,args=(a_flag,)).start()
                        return
                    ret, frame = cap.read()
                    # print("cap.read()-->", ret)
                    cur = cap.get(1)  # 获取当前帧数
                    if ret:
                        # self.window.update_video_stream(frame)
                        # byte_arrays = pickle.dumps(frame)
                        config_util.frame.append(frame)
                        self.window.video_signal.emit(QByteArray())
                        cv2.waitKey(1)  # 等待1毫秒 （1秒=1000毫秒）
                    sleepTime = (e + cur) / rate - time.time() + startTime
                    if sleepTime > 0:  # 播放时间快了就等一下
                        time.sleep(sleepTime)
                    if not ret:
                        break
                e += cur

    # 根据视频的帧率来刷新文字
    def answer_text_single(self,a_flag):
        if self.textlist:
            # print("vedio windows 411 textlist:", self.textlist)
            text ,total_time= self.textlist.popleft()
            self.window.update_listening_dianzan(True)
            self.window.update_listening_cai(True)
            self.window.update_listening_budianzan(False)
            self.window.update_listening_bucai(False)
            # 将total_time 分配给文字的长度
            text_len = 5
            total_txt=""
            txtlist = [text[i:i + text_len] for i in range(0, len(text), text_len)]
            # print("vedio windows 411 textlist (你好之前):", txtlist)
            if a_flag == "你好":
                self.zhanshitext = ""

            # print("vedio windows 411 zhanshitext (list之前):", self.zhanshitext)

            print("vedio windows 411 text:", text)
            # print("vedio windows 411 flag:", a_flag)
            # print("vedio windows 411 textlist:", txtlist)
            # self.window.answer_signal.emit(self.zhanshitext)
            print("--txtlist->",a_flag,config_util.get_last_question())
            if a_flag != config_util.get_last_question() and not self.is_think:
                # self.textlist.clear()
                # self.zhanshitext = ""
                return

            stime = total_time / len(text)
            # print("vedio windows 411 zhanshitext (循环之前):", self.zhanshitext)
            # print("vedio windows 411 textlist (循环之前):", txtlist)
            for txt in txtlist:

                if a_flag != config_util.get_last_question() and not self.is_think:
                    return
                self.zhanshitext =self.zhanshitext + txt
                # print("vedio windows 411 zhanshitext:", self.zhanshitext)
                #展板展示的信息
                self.window.answer_signal.emit(self.zhanshitext)
                # self.window.answer_signal.emit("测试")

                stime2 = stime*(len(txt)-1)
                if stime >0:
                    time.sleep(stime2)

            if text == "嗯，容我想一想" or text == "我在有什么需要":
                self.zhanshitext = ""


            # while i < len(text):
            #     # 暂停功能
            #     if a_flag != config_util.get_last_question() and not self.is_think:
            #         return
            #     i += text_len
            #     self.window.answer_signal.emit(text[:i])
            #     time.sleep(stime)
            # self.zhanshitext += text

    # 在播放列表中存入元素
    def offer(self, mp3_path, frame_list, answer_text, mode, total_frame_size=None, segment_code=0,a_flag="",is_think = False):
        """
        参数：
            mp3_path: mp3的文件位置（目前仅支持本地文件）
            frame_list: 帧列表
            answer_text: 回答的文字信息（为None时不会刷新答案）
            mode: 缓存模式 *
            total_frame_size: [可选] 总共帧的张数
            is_segment: [可选] 是否为分段中的后面段数（主要用于判断是否需要播放音频，以及播放时预加载后续的视频）
        """
        print("--offer->",mp3_path,segment_code,a_flag,answer_text)
        self.playlist.append((mp3_path, frame_list, answer_text, mode,
                              total_frame_size if total_frame_size is not None else len(frame_list),
                              segment_code,a_flag,is_think))


class VideoWindow(QWidget):
    answer_signal = pyqtSignal(str)
    question_signal = pyqtSignal(str)
    video_signal = pyqtSignal(QByteArray)
    gif_signal = pyqtSignal(QByteArray)

    pause_signal = pyqtSignal(bool)
    update_signal = pyqtSignal(str)
    update_signal_shuziren = pyqtSignal(str)
    start_signal = pyqtSignal(bool)
    flowLayout = None
    shuziren = None
    def __init__(self):
        super().__init__()
        self.ui = None
        self.pause_flag = False
        self.init_ui()

    def do_speak(self,question):
        # if config_util.video_stream.playing == True:
        config_util.set_last_question("")
            # time.sleep(0.5)

        config_util.video_stream.playing = True
        # self.change_pause_button(False)
        # config_util.config["interact"]["playSound"] = False
        # config_util.save_config(config_util.config)
        config_util.set_last_question(question["question"])
        self.question_signal.emit(question["question"])  # 更新问题
        fay_booter.speak(question["question"], question["answer"])
        # fay_booter.speak("你好", question["answer"])
        # self.update_listening_tingx(True)
        self.update_listening_dengdaix(True)
        # self.update_listening_shuox(False)
        # self.update_question(question["question"])
        # self.update_answer(question["answer"])
    def button_clicked(self,question):
        # self.ui.label_10.setHidden(True)
        thread = threading.Thread(target=self.do_speak, args=(question,))
        thread.start()

    def show_close(self,show):
        if show == True:
            self.ui.label_6.setHidden(show)
            self.ui.label_7.setHidden(show)
            self.ui.label_8.setHidden(show)
            self.ui.label_9.setHidden(show)
            self.ui.label_10.setHidden(show)
            self.ui.label_11.setHidden(show)
        self.ui.label_13.setHidden(show)
        self.ui.close.raise_()
        self.ui.close.setHidden(not show)

    def init_ui(self):
        # # 创建按钮
        # button1 = QLabel("Button 1")
        # button2 = QLabel("Button 2")
        # button3 = QLabel("Button 3")
        # button4 = QLabel("Button 4")
        #
        # # 创建水平流布局
        # hbox = QHBoxLayout()
        # hbox.addStretch(1)
        # hbox.addWidget(button1)
        # hbox.addStretch(1)
        # hbox.addWidget(button2)
        # hbox.addStretch(1)
        # hbox.addWidget(button3)
        # hbox.addStretch(1)
        # hbox.addWidget(button4)
        # hbox.addStretch(1)
        #
        # # 设置窗口的布局
        # self.setLayout(hbox)
        # self.setGeometry(100, 100, 10, 10)border-radius: 10px;padding: 10px 30px;

        # self.setLayout(flowLayout)
        # return
        if config_util.video_height == 3840:
            self.ui = uic.loadUi('gui/video_window_3840.ui')
        else:
            self.ui = uic.loadUi('gui/video_window.ui')
        self.ui.pushbutton1.setHidden(True)
        self.ui.yidianzan.setHidden(True)
        self.ui.yicai.setHidden(True)
        self.ui.yicai.setHidden(True)
        self.ui.label_6.setHidden(True)
        self.ui.label_7.setHidden(True)
        self.ui.label_11.setHidden(True)
        self.ui.label_8.setHidden(True)
        self.ui.label_9.setHidden(True)
        self.ui.label_10.setHidden(True)
        self.ui.label_14.setHidden(True)
        self.ui.label_13.setHidden(True)
        self.ui.dianzan.setHidden(True)
        self.ui.yidianzan.setHidden(True)
        self.ui.cai.setHidden(True)
        self.ui.yicai.setHidden(True)
        self.ui.label_12.setHidden(True)
        self.ui.close.setHidden(True)
        self.ui.close_2.setHidden(True)
        self.ui.shezhikai_3.setHidden(True)
#         html = markdown.markdown("""
# <style>
# body {
#     background-color: rgba(255, 255, 255, 0); /* 设置背景为透明 */
#     color: #000; /* 文本颜色为黑色，以便在透明背景上可见 */
# }
# </style>
#
# ## XPopup
# ![](https://api.bintray.com/packages/li-xiaojun/jrepo/xpopup/images/download.svg)  ![](https://img.shields.io/badge/platform-android-blue.svg)  ![](https://img.shields.io/badge/author-li--xiaojun-brightgreen.svg) ![](https://img.shields.io/badge/compileSdkVersion-28-blue.svg) ![](https://img.shields.io/badge/minSdkVersion-19-blue.svg) ![](https://img.shields.io/hexpm/l/plug.svg)
# ![](screenshot/logo.png)
#
# 国内Gitee镜像地址：https://gitee.com/lxj_gitee/XPopup
#
# ## 好站推荐
# 1. 全网价格最低的公众号和小程序微商城。官网：https://www.xingke.vip
#
#
# ### 中文 | [English](https://github.com/li-xiaojun/XPopup/blob/master/README-en.md)
# - 内置几种了常用的弹窗，十几种良好的动画，将弹窗和动画的自定义设计的极其简单；目前还没有出现XPopup实现不了的弹窗效果。
#   内置弹窗允许你使用项目已有的布局，同时还能用上XPopup提供的动画，交互和逻辑封装。
# - UI动画简洁，遵循Material Design，在设计动画的时候考虑了很多细节，过渡，层级的变化
# - 交互优雅，实现了优雅的手势交互，智能的嵌套滚动，智能的输入法交互，具体看Demo
# - 适配全面屏和各种挖孔屏，目前适配了小米，华为，谷歌，OPPO，VIVO，三星，魅族，一加全系全面屏手机
# - 自动监听Activity/Fragment生命周期或任意拥有Lifecycle的UI组件，自动释放资源。在Activity/Fragment直接finish的场景也避免了内存泄漏
# - XPopup实现了LifecycleOwner，可以直接被LiveData监视生命周期，弹窗可见时才更新数据，不可见不更新
# - 很好的易用性，自定义弹窗只需继承对应的类，实现你的布局，然后像Activity那样，在`onCreate`方法写逻辑即可
# - 性能优异，动画流畅；精心优化的动画，让你很难遇到卡顿场景
# - 支持在应用后台弹出（需要申请悬浮窗权限，一行代码即可）
# - 支持androidx，完美支持RTL布局，完美支持横竖屏切换，支持小窗模式
# - **如果你想要时间选择器和城市选择器，可以使用XPopup扩展功能库XPopupExt： https://github.com/li-xiaojun/XPopupExt**
#
#
#         """,extensions=['markdown.extensions.extra'])
#         self.ui.text_mark.setAttribute(Qt.WA_TranslucentBackground, True)
#         self.ui.text_mark.setStyleSheet("background: transparent;")
#         self.ui.text_mark.page().setBackgroundColor(Qt.GlobalColor.transparent)
#         self.ui.text_mark.setHtml(html)
        # 暂停按钮
        self.ui.pause.setHidden(True)

        self.ui.shezhikai.setHidden(True)
        self.ui.shezhiguan.setHidden(True)

        self.ui.mohu.setHidden(True)

        self.update_listening(True)
        self.setplaying = True
        self.cur_voice = None
        self.changes = False
        self.change_cache = False
        self.change_jingyin = False
        self.change_jingyins = False
        self.change_caches = False
        self.changess = False
        s_list = shuziren_db.get_list(0, 1)
        if len(s_list) == 0:
            self.shuziren = {"id":1,"name":"默认","sound":"zhifeng_emo","image":"","video":"/gui/static/model_man.mp4","video2":"/gui/static/train_model_man.mp4"}
        else:
            self.shuziren = s_list[0]
        # self.default = self.shuziren.video
        # self.default1 = f'resource/data/default/defaultPart001.mp3'
        # self.face = self.shuziren.video2
        # self.cache_path = f'cache_data/{self.shuziren.video.id}'
        # self.pkl_data_path = f'{self.cache_path}/model/model_{self.shuziren.video.id}.pkl'
        print("默认数字人",self.shuziren)
        self.change()
        # 隐藏标题栏
        self.ui.setWindowFlags(Qt.FramelessWindowHint)

        screen = tkinter.Tk()
        self.screen_width = screen.winfo_screenwidth()
        self.screen_height = screen.winfo_screenheight()
        print(f"[info] 当前屏幕的分辨率是：{self.screen_width} x {self.screen_height}")
        # 关闭窗口
        screen.destroy()

        # 调整界面大小
        self.ui.resize(self.screen_width, self.screen_height)
        # 调整视频大小
        self.ui.video_show.resize(self.screen_width, self.screen_height)

        # 调整字体
        font = QFont("黑体", 30)
        self.ui.question_text.setFont(font)
        self.ui.answer_text.setFont(font)

        # 绑定信号与槽
        self.answer_signal.connect(self.update_answer)
        self.question_signal.connect(self.update_question)
        self.video_signal.connect(self.update_video_stream)

        # 暂停函数
        self.ui.pause.clicked.connect(self.change_pause_button2)
        self.pause_signal.connect(self.change_pause_button)
        self.ui.pushbutton.clicked.connect(self.on_maikekai_clicked)
        self.ui.pushbutton1.clicked.connect(self.on_maikeguan_clicked)
        self.ui.close_no.clicked.connect(lambda: self.show_close(False))
        self.ui.close_yes.clicked.connect(lambda: os._exit(123))

        self.ui.pushbutton2.clicked.connect(lambda: self.show_close(True))
        self.ui.dianzan.clicked.connect(self.update_listening_dianzan1)
        self.ui.cai.clicked.connect(self.update_listening_diancai)

        # self.ui.shezhikai_3.clicked.connect(self.change)
        self.ui.shezhikai_2.clicked.connect(lambda : self.ui.close_2.setHidden(False))
        self.ui.close_no_4.clicked.connect(lambda: self.ui.close_2.setHidden(True))
        self.ui.close_yes_4.clicked.connect(self.change)
        # 隐藏垂直滚动条
        self.ui.answer_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.answer_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 隐藏垂直滚动条
        self.ui.question_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.question_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # print()


        # button2_size_policy = QSizePolicy()
        # button2_size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
        # button2_size_policy.setVerticalPolicy(QSizePolicy.Preferred)
        # self.ui.q_list.setSizePolicy(button2_size_policy)
        self.flowLayout = FlowLayout(self.ui.q_list,30,20)
        self.flowLayoutShuziren = FlowLayout(self.ui.shuziren, 30, 20)
        # self.flowLayout.setSpacing(30)
        # self.flowLayout.setContentsMargins(30, 30, 30, 30)
        # self.ui.q_list.setLayout(self.flowLayout)
        self.update_signal.connect(lambda: self.add_shuziren())
        self.update_signal_shuziren.connect(lambda: self.add_shuziren())
        self.start_signal.connect(self.update_listening_qidongs)

    def add_question_small(self):
        flowLayout = self.flowLayout
        len  = self.add_question(flowLayout)
        if(len>0):
            button = QPushButton('更多问题>>')
            button.setStyleSheet(
                "QPushButton {background-color: rgba(255, 255, 255, 76); color: white;   border: unset;border-radius: 10px;padding: 10px 30px;font-weight: 600; font-size: 34px;}"
                "QPushButton:pressed {background-color: rgba(255, 255, 255, 176); color: white;}"
            )

            def button_click_handler():
                return lambda: self.add_question(flowLayout, 0, 1000, True)

            button.clicked.connect(button_click_handler())
            flowLayout.addWidget(button)
    def add_shuziren(self,flowLayout=None,n=0,l=6,small=False):
        flowLayout = self.flowLayoutShuziren
        print("开始添加数字人")
        layout_items = [flowLayout.itemAt(i) for i in range(flowLayout.count())]
        for item in layout_items:
            widget = item.widget()
            if widget is not None:
                flowLayout.removeWidget(widget)
                widget.deleteLater()
            # 调整布局
            # button.deleteLater()
            # # 更新界面
            # self.update()
        q_a_list = shuziren_db.get_list(0, 1000)
        # config_util.q_a_list=q_a_list
        # if l == 6:
        #     q_list = q_a_list[:6]
        # else:
        q_list = q_a_list

        for question in q_list:
            print("--shuziren->",question['name'],id(question),id(self.shuziren),question)
            button = QPushButton(question['name'])
            if question['id'] == self.shuziren['id'] if self.shuziren is not None else 0:
                button.setStyleSheet(
                    "QPushButton {background-color: rgba(0, 0, 255, 76); color: white;   border: unset;border-radius: 10px;padding: 10px 30px;font-weight: 600; font-size: 34px;}"
                    "QPushButton:pressed {background-color: rgba(255, 255, 255, 176); color: white;}"
                )
            else:
                button.setStyleSheet(
                    "QPushButton {background-color: rgba(255, 255, 255, 76); color: white;   border: unset;border-radius: 10px;padding: 10px 30px;font-weight: 600; font-size: 34px;}"
                    "QPushButton:pressed {background-color: rgba(255, 255, 255, 176); color: white;}"
                )

            def do_samll(q,flowLayout):
                self.shuziren = q
                print("选择数字人：",id(q),"<>",id(self.shuziren))
                self.add_shuziren(flowLayout)
                # if(small):
                #     self.add_question_small()
                # self.button_clicked(q)
            def button_click_handler(q,flowLayout):
                return lambda: do_samll(q,flowLayout)

            button.clicked.connect(button_click_handler(question,flowLayout))
            flowLayout.addWidget(button)
        return len(q_list)
    def add_question(self,flowLayout,n=0,l=6,small=False):

        layout_items = [flowLayout.itemAt(i) for i in range(flowLayout.count())]
        for item in layout_items:
            widget = item.widget()
            if widget is not None:
                flowLayout.removeWidget(widget)
                widget.deleteLater()
            # 调整布局
            # button.deleteLater()
            # # 更新界面
            # self.update()
        q_a_list = question_db.get_list(0, 1000)
        config_util.q_a_list=q_a_list
        if l == 6:
            q_list = q_a_list[:6]
        else:
            q_list = q_a_list

        for question in q_list:
            button = QPushButton(question['question'])
            button.setStyleSheet(
                "QPushButton {background-color: rgba(255, 255, 255, 76); color: white;   border: unset;border-radius: 10px;padding: 10px 30px;font-weight: 600; font-size: 34px;}"
                "QPushButton:pressed {background-color: rgba(255, 255, 255, 176); color: white;}"
            )

            def do_samll(q):
                if(small):
                    self.add_question_small()
                self.button_clicked(q)
            def button_click_handler(q):
                return lambda: do_samll(q)

            button.clicked.connect(button_click_handler(question))
            flowLayout.addWidget(button)
        return len(q_list)

    def update_video_stream(self, frame):
        try:
            # 可能会引发异常的代码块

        # 将帧转换为QImage
        #     frame = pickle.loads(frame)
            frame = config_util.frame.popleft()
            image = QImage(frame.data,
                           frame.shape[1],
                           frame.shape[0],
                           frame.strides[0],
                           QImage.Format_RGB888).rgbSwapped()

            # 将图像调整为小部件的大小
            # scaled_image = image.scaled(self.ui.video_show.size(),
            #                             Qt.AspectRatioMode.IgnoreAspectRatio)
            # print("正在放视频")
            # 在标签上显示图像
            self.ui.video_show.setPixmap(QPixmap.fromImage(image))
        except Exception as e:
            # 捕获所有其他类型的异常
            print(f"发生了异常: {e}")


    def change(self):
        config_util.last_question=""
        shuziren = self.shuziren
        # self.cur_voice = 'zhimiao_emo'
        self.cur_voice = shuziren['sound']
        # self.default = f'resource/data/default/default_model_lady.mp4'
        self.default = 'gui'+shuziren['video']
        # self.default1 = f'resource/data/default/defaultPart0012.mp3'
        self.default1 = None
        # self.face = 'resource/data/train_model_lady.mp4'
        self.face = 'gui/'+shuziren['video2']
        # self.pkl_data_path = 'resource/data/model_lady.pkl'

        self.cache_path = f'cache_data/{shuziren["id"]}'
        self.pkl_data_path = f'{self.cache_path}/model/model_{shuziren["id"]}.pkl'
        for folder in [self.cache_path+'/mp4',self.cache_path+'/mp3',self.cache_path+'/mp4_temp',self.cache_path+'/mp3_temp',self.cache_path+'/text_temp',f'{self.cache_path}/model',f'{self.cache_path}/audio']:
            try:
                os.makedirs(folder, exist_ok=True)
                # print(f"Folder {folder} created successfully.")
            except Exception as e:
                print(f"Failed to create folder {folder}: {e}")
        threading.Thread(target=inference_torch.chenge).start()
        self.change_jingyin = True
        self.change_cache = True
        self.changes = True
        print("更改人物",shuziren)


    def change_nan(self):
        self.cur_voice = 'zhifeng_emo'

        self.default = f'resource/data/default/model_man.mp4'


        self.default1 = f'resource/data/default/defaultPart001.mp3'


        self.face = 'resource/data/train_model_man.mp4'
        self.pkl_data_path = 'resource/data/model_man.pkl'

        self.cache_path = 'cache_data_one'
        threading.Thread(target=inference_torch.chenge).start()
        self.change_jingyins = True
        self.change_caches = True
        self.changess = True
        print("更改声音")

    # 显示暂停重置的函数
    def update_pause(self, show):
        # update_video_stream
        self.ui.pause.setHidden(show)

    # 正在说话，点击暂停
    def change_pause_button(self,show = False):
        print("pause_flag--->",show)
        # if show == True:
        #     try:
        #         self.thread_answer._stop()
        #         self.thread_video._stop()
        #         print("成功执行thread._stop()，线程已停止（若可停止）")
        #     except Exception:
        #         print(Exception)
        self.pause_flag = show
        # if show == True:
        #     self.pause_flag = show
            # timer = threading.Timer(1, self.change_pause_button)
            # timer.start()

    def change_pause_button2(self,show):
        print("pause_flag--->",True)
        config_util.set_last_question("")
        # if show == True:
        #     try:
        #         self.thread_answer._stop()
        #         self.thread_video._stop()
        #         print("成功执行thread._stop()，线程已停止（若可停止）")
        #     except Exception:
        #         print(Exception)
        self.pause_flag = True

    def update_listening(self, show):
        pass
        # self.ui.listen_label.setHidden(show)

    def update_listening_shezhigkai(self, show):
        self.ui.shezhikai.setHidden(show)

    def update_listening_qidongs(self, show):
        pass
        if show == True:
            self.ui.label_13.setHidden(False)
            self.add_question_small()
            self.add_shuziren()
        else:
            self.ui.label_13.setHidden(True)
        # self.ui.label_3.setHidden(show)

    def update_listening_qidongx(self, show):
        self.ui.label_6.setHidden(show)

    def update_listening_dengdaix(self, show):
        if config_util.video_stream.playing:
            self.ui.label_7.setHidden(True)
        else:
            self.ui.label_7.setHidden(show)
        if not self.ui.label_11.isHidden():
            self.ui.label_7.setHidden(True)


    def update_listening_dengdais(self, show):
        pass
        # self.ui.label_5.setHidden(show)

    def update_listening_tings(self, show):
        pass
        # self.ui.label_4.setHidden(show)

    def update_listening_tingx(self, show):
        self.ui.label_10.setHidden(show)

    def update_listening_shuos(self, show):
        pass
        # self.ui.label_2.setHidden(show)

    def update_listening_shuox(self, show):
        # 这是正在讲话
        self.ui.label_9.setHidden(show)

    def update_listening_sikao(self, show):
        self.ui.label_11.setHidden(show)
        if not show:
            self.ui.label_7.setHidden(True)

    def update_listening_dianzan(self, show):
        self.ui.yidianzan.setHidden(show)

    def update_listening_budianzan(self, show):
        self.ui.dianzan.setHidden(show)

    def update_listening_cai(self, show):
        self.ui.yicai.setHidden(show)

    def update_listening_bucai(self, show):
        self.ui.cai.setHidden(show)

    def update_listening_duihuakuang1(self, show):
        # self.ui.label_13.setHidden(show)
        self.ui.label_12.setHidden(show)

    def update_listening_duihuakuang2(self, show):
        self.ui.label_14.setHidden(show)

    def update_listening_dianzan1(self):
        self.ui.dianzan.setHidden(True)
        self.ui.yidianzan.setHidden(False)
        self.ui.cai.setHidden(False)
        self.ui.yicai.setHidden(True)

    def update_listening_diancai(self):
        self.ui.cai.setHidden(True)
        self.ui.yicai.setHidden(False)
        self.ui.dianzan.setHidden(False)
        self.ui.yidianzan.setHidden(True)

    # def update_listening_qidong2(self):
    #     self.ui.answer_text.setHidden(False)
    # 更新答案文本
    def update_answer(self, text):
        self.ui.answer_text.setText(markdown.markdown(text))
        # self.ui.answer_text.setText(text)
        self.ui.answer_text.verticalScrollBar().setValue(self.ui.answer_text.verticalScrollBar().maximum())

    # 更新问题文本
    def update_question(self, text):
        # 每次可以提问问题的时候就把暂停按钮释放掉  回复最初状态
        self.ui.question_text.setText(text)
        self.ui.question_text.verticalScrollBar().setValue(self.ui.question_text.verticalScrollBar().maximum())

    def on_maikekai_clicked(self):
        # config_util.config["interact"]["playSound"] = True
        # config_util.save_config(config_util.config)
        config_util.mute=True
        config_util.set_last_question("")
        self.setplaying = False
        self.ui.label_6.setHidden(True)
        self.ui.label_7.setHidden(True)
        self.ui.label_11.setHidden(True)
        # self.ui.label_8.setHidden(False)
        self.ui.label_9.setHidden(True)
        self.ui.label_10.setHidden(True)
        self.ui.pause.setHidden(True)
        print("mic 开")

    def on_maikeguan_clicked(self):
        # config_util.config["interact"]["playSound"] = False
        # config_util.save_config(config_util.config)
        config_util.mute=False
        self.setplaying = True
        self.ui.label_7.setHidden(False)
        self.ui.pause.setHidden(False)
        print("mic 关")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setWindowIcon(QtGui.QIcon('icon.png'))

    # MainWindow()

    video_window = VideoWindow()
    video_window.show()

    # video_stream = VideoStream(video_window)
    # config_util.video_stream = video_stream
    # video_stream.start()


    sys.exit(app.exec_())
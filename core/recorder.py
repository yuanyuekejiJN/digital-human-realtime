import audioop
import datetime
import math
import threading
import time
import wave
from abc import abstractmethod
import concurrent.futures


from ai_module.ali_nls import ALiNls
from ai_module.funasr import FunASR
from ai_module.xf_nls import XfNls
from core import wsa_server
from scheduler.thread_manager import MyThread
from utils import util, config_util
from utils import config_util as cfg
import numpy as np

# 启动时间 (秒)
_ATTACK = 0.001

# 释放时间 (秒)
_RELEASE = 0.75
started = False

class Recorder:

    def __init__(self, fay):
        self.__fay = fay

        # 正在运行标志位
        self.__running = True

        # 正在处理标志位
        self.__processing = False
        self.__history_level = []
        self.__history_data = []
        self.__dynamic_threshold = 0  # 声音识别的音量阈值

        self.__MAX_LEVEL = 25000
        self.__MAX_BLOCK = 100
        self.channels = 1
        # Edit by xszyou in 20230516:增加本地asr
        self.ASRMode = cfg.ASR_mode
        self.__aLiNls = None  # 获取语音转文字的实例对象
        self.video_stream = config_util.video_stream
        self.sample_rate = 16000


    # 返回语音转文字的对象
    def asrclient(self,audio_id):
        if self.ASRMode == "ali":
            asrcli = ALiNls(audio_id=audio_id)
        elif self.ASRMode == "keda":
            asrcli = XfNls()
        elif self.ASRMode == "funasr":
            asrcli = FunASR()
        return asrcli

    # 计算声音均方根值前number个的平均值
    def __get_history_average(self, number):
        return sum(self.__history_level[::-1][:number]) / min(max(len(self.__history_level), 1), number)

    # 根据level的特定计算公式，计算历史记录的百分率
    def __get_history_percentage(self, number):
        return (self.__get_history_average(number) / self.__MAX_LEVEL) * 1.05 + 0.02

    def __print_level(self, level):
        text = ""
        per = level / self.__MAX_LEVEL
        if per > 1:
            per = 1
        bs = int(per * self.__MAX_BLOCK)
        for i in range(bs):
            text += "#"
        for i in range(self.__MAX_BLOCK - bs):
            text += "-"
        print(text + " [" + str(int(per * 100)) + "%]")

    # 等待处理结果
    def __waitingResult(self, iat: asrclient):
        if self.__fay.playing:
            return
        self.processing = True
        t = time.time()
        tm = time.time()
        # 等待结果返回 #and time.time() - t < 1
        while not iat.done and time.time() - t < 5:
            time.sleep(0.01)
        print("iat.done:",iat.done,time.time() - t)
        text = iat.finalResults
        util.log(1, "语音处理完成！ 耗时: {} ms".format(math.floor((time.time() - tm) * 1000)))
        # self.video_stream.window.update_listening_tingx(True)
        if len(text) > 0:
            # 语音识别完毕，处理识别到的文本信息
            self.on_speaking(text)
            self.processing = False
            # self.video_stream.window.update_listening_tingx(True)
            # self.video_stream.window.update_listening_dengdaix(False)

        else:
            util.log(1, "[!] 语音未检测到内容！")
            # self.video_stream.window.update_listening_tingx(True)
            # self.video_stream.window.update_listening_dengdaix(False)
            self.processing = False
            self.dynamic_threshold = self.__get_history_percentage(30)
            wsa_server.get_web_instance().add_cmd({"panelMsg": ""})
            if not cfg.config["interact"]["playSound"]:  # 非展板播放
                content = {'Topic': 'Unreal', 'Data': {'Key': 'log', 'Value': ""}}
                wsa_server.get_instance().add_cmd(content)

    def textSimulation(self, text):
        iat = XfNls(text)
        iat.done = True
        # iat = self.asrclient(text)
        self.__waitingResult(iat)
        # iat.end()

    # 线程：
    def __record(self):
        # 1. 获取声音的流
        try:
            stream = self.get_stream()  # 把get stream的方式封装出来方便实现麦克风录制及网络流等不同的流录制子类
        except Exception as e:
            print(e)
            util.log(1, "请检查设备是否有误，再重新启动!")
            return
        self.isSpeaking = False
        last_mute_time = 0
        self.last_speaking_time = time.time()
        data = None
        self.audio_data_list = []
        self.audio_id = ""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        while self.__running:
            # 1. 从stream中读取1024个字节
            try:
                data = stream.read(512, exception_on_overflow=False)
                if config_util.mute:
                    continue
                # print("data-length->",len(data))
            except Exception as e:
                data = None
                print(e)
                util.log(1, "请检查设备是否有误，再重新启动!")
                return

            if data is None:
                continue

            # # 2. 根据配置文件的开关决定channels
            # if cfg.config['source']['record']['enabled']:
            #     if len(cfg.config['source']['record']) < 3:
            #         channels = 1
            #     else:
            #         channels = int(cfg.config['source']['record']['channels'])
            #
            #     # 只获取第一声道
            #     data = np.frombuffer(data, dtype=np.int16)
            #
            #     # BRONCOS: 此处需要了解np的具体用法
            #     data = np.reshape(data, (-1, channels))  # reshaping the array to split the channels
            #     # 取data数组的第一维的第0个数据，重新组成一个一维数组赋值给mono。
            #     mono = data[:, 0]  # taking the first channel
            #     data = mono.tobytes()

            # level 记录声音的均方根值
            level = audioop.rms(data, 2)
            # 记录历史data和历史level，data最高记录5条，level最多记录500条，作用未知
            if len(self.__history_data) >= 10:
                self.__history_data.pop(0)
            if len(self.__history_level) >= 500:
                self.__history_level.pop(0)
            self.__history_data.append(data)
            self.__history_level.append(level)

            # 应该是计算当前音量
            percentage = level / self.__MAX_LEVEL
            history_percentage = self.__get_history_percentage(30)

            if history_percentage > self.__dynamic_threshold:
                self.__dynamic_threshold += (history_percentage - self.__dynamic_threshold) * 0.0025
            elif history_percentage < self.__dynamic_threshold:
                self.__dynamic_threshold += (history_percentage - self.__dynamic_threshold) * 1

            soon = False

            '''
            逻辑：
                1. 判断是否可以聆听当前声音
                2. 消耗记录的声音数据
                3. 发送给nls转文字
            '''
            # print("音量->", level,percentage,self.__dynamic_threshold)
            # print(self.__fay.video_stream, self.__fay.video_stream.playing)
            # print("音量->",percentage)
            # print("音量->",percentage > self.__dynamic_threshold,percentage,"<-->",self.__dynamic_threshold)
            executor.submit(self.do_spreaking, data,percentage)
    def do_spreaking(self,data,percentage):
        if percentage > self.__dynamic_threshold:
            # 上次说话的时间
            self.last_speaking_time = time.time()

            # 条件1：暂时未用
            # 条件2：
            # 条件3：距离上次静音时间需要超过启动时间（缓冲时间）
            # if not self.__processing and not isSpeaking and time.time() - last_mute_time < _ATTACK:
            #     print("浪费的声音")
            if not self.__processing and not self.isSpeaking:

                self.isSpeaking = True  # 用户正在说话
                self.video_stream.window.update_listening_dengdaix(True)
                self.video_stream.window.update_listening_tingx(False)

                util.log(3, "聆听中...")
                now = datetime.datetime.now()
                self.audio_id = now.strftime('%m-%d-%H-%M-%S-%f', )
                # 拿到语音转文字的实例对象
                self.__aLiNls = self.asrclient(self.audio_id)

                try:
                    # 连接nls，做准备工作（准备头信息、token等）
                    self.__aLiNls.start()
                    # t = time.time()
                    # while not self.__aLiNls.started and time.time() - t < 2:
                    #     time.sleep(0.01)
                    print("--ali-start->", self.audio_id)
                except Exception as e:
                    print("-ali-nls-eror->", e)
                for i in range(len(self.__history_data) - 1):  # 当前data在下面会做发送，这里是发送激活前的音频数据，以免漏掉信息
                    buf = self.__history_data[i]
                    self.audio_data_list.append(self.__process_audio_data(buf, self.channels))
                    if self.ASRMode == "ali":
                        self.__aLiNls.send(self.__process_audio_data(buf, self.channels).tobytes())
                    else:
                        pass
                self.__history_data.clear()

        else:
            # last_mute_time = time.time()
            if self.isSpeaking and time.time() - self.last_speaking_time > _RELEASE:
                self.isSpeaking = False
                # 设置播放器状态
                self.__fay.video_stream.wait_playing = True
                self.__fay.video_stream.window.update_listening(True)

                # self.__fay.video_stream.window.update_listening_sikao(False)

                self.__aLiNls.end()
                self.video_stream.window.update_listening_tingx(True)
                self.video_stream.window.update_listening_dengdaix(False)
                util.log(1, "语音处理中...")
                # self.__fay.last_quest_time = time.time()
                self.__waitingResult(self.__aLiNls)
                # threading.Thread(target=self.__waitingResult, args=(self.__aLiNls,)).start()
                # self.video_stream.window.update_listening_tingx(True)
                mono_data = self.__concatenate_audio_data(self.audio_data_list)
                # self.__save_audio_to_wav(mono_data, self.sample_rate, 'cache_data/input'+str(time.time())+'.wav')
                threading.Thread(target=self.__save_audio_to_wav,
                                 args=(mono_data, self.sample_rate, f'cache_data/{self.video_stream.window.shuziren["id"]}/audio/' + self.audio_id + 'input1.wav')).start()
                self.audio_data_list = []

        if self.isSpeaking:
            self.audio_data_list.append(self.__process_audio_data(data, self.channels))
            self.__aLiNls.send(self.__process_audio_data(data, self.channels).tobytes())

    def __save_audio_to_wav(self, data, sample_rate, filename):
        # 确保数据类型为 int16
        if data.dtype != np.int16:
            data = data.astype(np.int16)

        # 打开 WAV 文件
        with wave.open(filename, 'wb') as wf:
            # 设置音频参数
            n_channels = 1  # 单声道
            sampwidth = 2  # 16 位音频，每个采样点 2 字节
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())

    def __concatenate_audio_data(self, audio_data_list):
        # 将累积的音频数据块连接起来
        data = np.concatenate(audio_data_list)
        return data
    # 转变为单声道np.int16
    def __process_audio_data(self, data, channels):
        data = bytearray(data)
        # 将字节数据转换为 numpy 数组
        data = np.frombuffer(data, dtype=np.int16)
        # 重塑数组，将数据分离成多个声道
        data = np.reshape(data, (-1, channels))
        # 对所有声道的数据进行平均，生成单声道
        mono_data = np.mean(data, axis=1).astype(np.int16)
        return mono_data


    def set_processing(self, processing):
        self.__processing = processing

    def start(self):
        global started
        if not started:
            started = True
            MyThread(target=self.__record).start()

    def stop(self):
        global started
        started = False
        self.__running = False
        self.__aLiNls.end()

    @abstractmethod
    def on_speaking(self, text):
        pass

    # TODO Edit by xszyou on 20230113:把流的获取方式封装出来方便实现麦克风录制及网络流等不同的流录制子类
    @abstractmethod
    def get_stream(self):
        pass

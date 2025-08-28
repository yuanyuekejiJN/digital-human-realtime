# from queue import Queue
#
# import pygame
#
#
# class SoundPlayer:
#     def __init__(self):
#         # 初始化pygame的混音器模块
#         pygame.mixer.init()
#         self.sound = Queue()
#
#     def load_sound(self, file,a_flag):
#         # 加载音频文件
#         self.sound[a_flag] = pygame.mixer.Sound(file)
#
#     def play_sound(self,a_flag):
#         for key in list(self.sound.keys()):
#             if key != a_flag:
#                 if self.sound[key]:
#                     self.sound[key].stop()
#         # 播放音频
#         sound = self.sound.get(a_flag)
#         if sound:
#             sound.stop()
#             sound.play()
#
#     def stop_sound(self,a_flag):
#         # 停止音频播放
#         sound = self.sound.get(a_flag)
#         if sound:
#             sound.stop()
#             try:
#                 self.sound.pop(a_flag)
#             except BaseException as e:
#                 print("stop_sound",e)
#             return True
#         return False
#
import pygame
import threading
import queue
import time

from utils import config_util


class SoundPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.audio_queue = queue.Queue()
        self.play_thread = None
        self.stop_event = threading.Event()
        self.play_sound()

    def play_sound(self):
        self.play_thread = threading.Thread(target=self.play_sound_real)
        self.play_thread.start()

    def play_sound_real(self):
        # while not self.stop_event.is_set():
            # if not self.audio_queue.empty():
        while True:
            a_flag,self.sound = self.audio_queue.get()
            try:
                # print("-->sound.play():")
                self.sound.play()
                config_util.video_stream.start_txt(a_flag)
                # 等待音频播放完成
                while pygame.mixer.get_busy():
                    time.sleep(0.01)
                # print("-->sound.play()2:")
                # self.audio_queue.task_done()
                # sound.play()
            except Exception as e:
                print(f"Error playing audio: {e}")
                # self.audio_queue.task_done()
            # time.sleep(0.01)

    def load_sound(self, audio_path, a_flag):
        # self.a_flag = a_flag
        sound = pygame.mixer.Sound(audio_path)
        if a_flag != config_util.last_question:
            return
        self.audio_queue.put((a_flag,sound))
        print(f"Added {audio_path} to queue")

    def stop_sound(self, a_flag):
        # pass
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.stop_event.set()
        self.sound.stop()
        # 等待音频停止
        # while pygame.mixer.get_busy():
        #     time.sleep(0.01)
        self.stop_event.clear()
        return True
        # 清除音频队列
        # with self.audio_queue.mutex:
        #     self.audio_queue.queue.clear()
        # self.start_play_thread()


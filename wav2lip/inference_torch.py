import argparse
import copy
import gc
import math
import os
import platform
import subprocess
import sys
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import as_completed

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import threading
import queue
import wave
import datetime

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wav2lip.torchalign import FacialLandmarkDetector
import wav2lip.audio
import pickle
from wav2lip.utils import decompose_tfm, img_warp, img_warp_back_inv_m, metrix_M
from wav2lip.utils import laplacianSmooth

from utils import config_util

torch.manual_seed(1234)

lock = threading.Lock()  # 创建一个锁对象

warnings.filterwarnings('ignore')
mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint



def load_model(path):
    from wav2lip.models.wav2lip import Wav2Lip
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


class Runner():
    def __init__(self, args, yolo_path, facial_path, data_path, video_stream):
        # 初始化
        self.gen_frame_num = None
        self.frame_list_res = None
        self.wav_path = None
        self.segment_data = (0, 0)  # (level, pos)

        self.segment_mode = config_util.segment_mode
        self.segment_basis = config_util.segment_basis

        self.build_frame_thread_size = config_util.build_frame_thread_size

        self.face_det = YOLO(yolo_path)

        lmk_net = FacialLandmarkDetector(facial_path)
        lmk_net = lmk_net.to(device)
        self.lmk_net = lmk_net.eval()

        self.device = device

        self.args = args
        self.pads = args.pads

        self.checkpoint_path = args.checkpoint_path
        self.batch_size = args.wav2lip_batch_size

        self.img_size = (256, 256)
        self.fps = 25

        self.a_alpha = 1.25
        self.audio_smooth = args.audio_smooth

        self.kpts_smoother = None
        self.abox_smoother = None

        # 拉普拉斯金字塔融合图片大小
        self.lpb_size = 256

        self.model = load_model(self.checkpoint_path)
        print("Model loaded")

        # 预加载推理pkl文件
        self.avatar = self.build_avatar(self.args.pkl_data_path, self.args.face, self.fps, self.args)
        print('Data preprocess loaded')

        # 设置音频缓存目录 主要用于其他格式的音频文件转换为wav文件
        self.temp_audio_path = os.path.join(data_path, 'temp', 'result.wav')
        c = time.time()
        wav = wav2lip.audio.load_wav("resource/think1.wav", 16000)
        d = time.time()
        print('预处理音频耗时：', d - c)

        self.video_stream = video_stream

    @staticmethod
    def landmark_to_keypoints(landmark):
        lefteye = np.mean(landmark[60:68, :], axis=0)
        righteye = np.mean(landmark[68:76, :], axis=0)
        nose = landmark[54, :]
        leftmouth = (landmark[76, :] + landmark[88, :]) / 2
        rightmouth = (landmark[82, :] + landmark[92, :]) / 2
        return (lefteye, righteye, nose, leftmouth, rightmouth)

    @torch.no_grad()
    def detect_face(self, face_img):
        boxes = self.face_det(face_img,
                              imgsz=640,
                              conf=0.01,
                              iou=0.5,
                              half=True,
                              augment=False,
                              device=self.device)[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        return bboxes

    @torch.no_grad()
    def detect_lmk(self, image, bbox=None):
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        bbox_tensor = torch.from_numpy(bbox[:, :4])
        landmark = self.lmk_net(img_pil, bbox=bbox_tensor, device=self.device).cpu().numpy()
        return landmark

    def prepare_batch(self, img_batch, mel_batch, img_size):
        img_size_h, img_size_w = img_size
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_batch = img_batch / 255.

        img_masked = img_batch.copy()
        img_masked[:, img_size_h // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3)
        # (B, 80, 16) -> (B, 80, 16, 1)
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        return img_batch, mel_batch

    def build_avatar(self, processor_data_path, video_path, fps, args, max_frame_num=-1):
        window = config_util.video_stream.window
        config_util.mute=True
        window.start_signal.emit(False)
        window.update_listening_qidongx(False)
        window.update_listening_dengdais(True)
        window.update_listening_dengdaix(True)
        window.update_listening_duihuakuang1(True)
        window.update_listening_duihuakuang2(True)
        window.update_listening_budianzan(True)
        window.update_listening_bucai(True)
        # 暂停按钮显示出来
        window.update_pause(True)

        # 该预处理文件不存在则开始进行处理
        if not os.path.exists(processor_data_path):
            # 预处理结果
            full_frames = []
            if video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
                full_frames = [cv2.imread(video_path)]
            else:
                video_stream = cv2.VideoCapture(video_path)
                fps = video_stream.get(cv2.CAP_PROP_FPS)
                self.fps = fps
                print("fps={}".format(fps))
                print('Reading video frames...')

                while 1:
                    still_reading, frame = video_stream.read()
                    if not still_reading:
                        video_stream.release()
                        break
                    if args.resize_factor > 1:
                        frame = cv2.resize(frame,
                                           (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

                    if args.rotate:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    y1, y2, x1, x2 = args.crop
                    if x2 == -1: x2 = frame.shape[1]
                    if y2 == -1: y2 = frame.shape[0]

                    frame = frame[y1:y2, x1:x2]

                    full_frames.append(frame)

                    if max_frame_num > 0 and len(full_frames) >= max_frame_num or args.static:
                        video_stream.release()
                        break

            print("Number of frames available for inference: " + str(len(full_frames)))

            self.kpts_smoother = laplacianSmooth()
            self.abox_smoother = laplacianSmooth()

            frame_info_list = []
            for frame_id in tqdm(range(len(full_frames))):
                imginfo = self.get_input_imginfo(full_frames[frame_id].copy())
                frame_info_list.append(imginfo)

            self.kpts_smoother = None
            self.abox_smoother = None

            frame_h, frame_w = full_frames[0].shape[:2]
            avatar = {
                'fps': fps,
                'frame_num': len(full_frames),
                'frame_h': frame_h,
                'frame_w': frame_w,
                'frame_info_list': frame_info_list
            }

            with open(processor_data_path, "wb") as file:
                pickle.dump(avatar, file)
            print('视频预处理结果存储成功，数据加载完成！path:{}'.format(processor_data_path))
            # return avatar

        # 从本地读取处理结果
        else:
            with open(processor_data_path, "rb") as file:
                avatar = pickle.load(file)
                # return avatar
        config_util.mute=False
        window.start_signal.emit(True)
        window.update_listening_qidongx(True)
        window.update_listening_dengdais(False)
        window.update_listening_dengdaix(False)
        window.update_listening_duihuakuang1(False)
        window.update_listening_duihuakuang2(False)
        window.update_listening_budianzan(False)
        window.update_listening_bucai(False)
        # 暂停按钮显示出来
        window.update_pause(False)
        return avatar

    @torch.no_grad()
    def get_input_imginfo(self, frame):
        bbox = self.detect_face(frame.copy())[0][:5]
        landmark = self.detect_lmk(frame.copy(), [bbox])[0]
        keypoints = self.landmark_to_keypoints(landmark)

        keypoints = self.kpts_smoother.smooth(np.array(keypoints))

        m = metrix_M(face_size=200, expand_size=256, keypoints=keypoints)

        align_frame = img_warp(frame, m, 256, adjust=0)
        align_bbox = self.detect_face(align_frame.copy())[0][:4]

        align_bbox = self.abox_smoother.smooth(np.reshape(align_bbox, (-1, 2))).reshape(-1)

        # 重新warp 图片，保持scale 不变
        w, h = 256, 256
        rt, s = decompose_tfm(m)
        s_x, s_y = s[0][0], s[1][1]
        m = rt
        align_frame = cv2.warpAffine(frame, m, (math.ceil(w / s_x), math.ceil(h / s_y)), flags=cv2.INTER_CUBIC)
        inv_m = cv2.invertAffineTransform(m)

        face = copy.deepcopy(align_frame)
        h, w, c = align_frame.shape
        bbox = align_bbox
        bbox[0] *= (w - 1) / 255
        bbox[1] *= (h - 1) / 255
        bbox[2] *= (w - 1) / 255
        bbox[3] *= (h - 1) / 255

        rect = [round(f) for f in bbox[:4]]
        pady1, pady2, padx1, padx2 = self.pads
        y1 = max(0, rect[1] - pady1)
        y2 = min(h, rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(w, rect[2] + padx2)

        coords = (y1, y2, x1, x2)
        face = face[y1:y2, x1:x2]

        face = cv2.resize(face, self.img_size)

        return {
            'img': face,
            'frame': frame,
            'coords': coords,
            'align_frame': align_frame,
            'm': m,
            'inv_m': inv_m,
        }

    def get_input_imginfo_by_index(self, idx, avatar):
        return avatar['frame_info_list'][idx]

    def get_input_mel_by_index(self, index, wav_mel):
        # 处理音频
        T = 5
        mel_idx_multiplier = 80. / self.fps  # 一帧图像对应3.2帧音频
        start_idx = int((index - (T - 1) // 2) * mel_idx_multiplier)
        if start_idx < 0:
            start_idx = 0
        if start_idx + mel_step_size > len(wav_mel[0]):
            start_idx = len(wav_mel[0]) - mel_step_size
        mel = wav_mel[:, start_idx: start_idx + mel_step_size]
        return mel

    def get_intput_by_index(self, index, wav_mel, avatar):
        mel = self.get_input_mel_by_index(index, wav_mel)
        index = config_util.lip_index
        config_util.lip_index = index+1
        # 处理图片，视频为正序，倒序，正序，倒序，循环
        frame_num = avatar['frame_num']
        idx = index % frame_num
        idx = idx if index // frame_num % 2 == 0 else frame_num - idx - 1

        input_dict = {'mel': mel}
        input_imginfo = self.get_input_imginfo_by_index(idx, avatar)
        input_dict.update(copy.deepcopy(input_imginfo))
        return input_dict

    def check_handle(self, cur, total):
        pos, flag = self.segment_data[1], False
        level = self.segment_data[0]

        mode = self.segment_mode
        basis = self.segment_basis

        # 0：不分段
        # 1：按帧数平均分段
        # 2：按比例平均分段（0~1之间）
        # 3：按比例不平均分配（0~1之间，最后值必须为1）
        if mode == 0:
            return 0, cur == total
        elif mode == 1:
            if cur >= (level + 1) * basis or cur == total:
                flag = True
                level += 1
                self.segment_data = (level, cur)
        elif mode == 2:
            if cur >= (level + 1) * basis * total:
                flag = True
                level += 1
                self.segment_data = (level, cur)
        elif mode == 3:
            if cur >= basis[level] * total:
                flag = True
                level += 1
                self.segment_data = (level, cur)

        return pos, flag

    def run(self, audio_path, content_text, play_mode,a_flag,is_think=False):
        # 初始化
        self.frame_list_res = []
        self.segment_data = (0, 0)

        args = self.args

        fps = args.fps
        self.fps = fps

        # 将其他音频格式是哦用ffmpeg转换为wav，此处可能会有问题
        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg  -loglevel error -y -i {} -strict -2 {}'.format(audio_path, self.temp_audio_path)
            subprocess.call(command, shell=True)
            wav_path = self.temp_audio_path
        else:
            wav_path = audio_path

        self.wav_path = wav_path
        # 处理音频
        c = time.time()
        wav = wav2lip.audio.load_wav(wav_path, 16000)
        wav_mel = wav2lip.audio.melspectrogram(wav)
        mel_idx_multiplier = 80. / fps

        # TODO: IMPORT: 生成视频帧的数量
        self.gen_frame_num = int(len(wav_mel[0]) / mel_idx_multiplier)
        d = time.time()
        print('处理音频耗时：', d - c)

        # 初始化时已预加载
        avatar = self.avatar
        if not avatar:
            return

        torch.cuda.empty_cache()

        batch_size = self.batch_size
        img_size = self.img_size

        batch_data = defaultdict(list)

        # 模型推断
        start_infer = time.time()
        frames_list = []  # 线程安全的队列

        # 创建一个线程池执行器  保证线程池只有一个线程
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        for i in tqdm(range(self.gen_frame_num)):
            # 暂停直接跳出去
            if a_flag != config_util.get_last_question() and not is_think:
                print("跳出去！不在加入新的视频")
                return

            input_data = self.get_intput_by_index(i, wav_mel, avatar)
            # 组batch
            for k, v in input_data.items():
                batch_data[k + '_batch'].append(v)

            if len(batch_data.get(
                    'mel_batch')) == batch_size or i == self.gen_frame_num - 1:
                infer_size = len(batch_data['mel_batch'])

                img_batch = batch_data['img_batch']
                mel_batch = batch_data['mel_batch']
                frames = batch_data['frame_batch']
                coords = batch_data['coords_batch']
                align_frames = batch_data['align_frame_batch']
                inv_ms = batch_data['inv_m_batch']

                if self.audio_smooth:
                    mel_batch.insert(0, self.get_input_mel_by_index(max(0, i - infer_size), wav_mel))
                    mel_batch.append(self.get_input_mel_by_index(min(i + 1, self.gen_frame_num - 1), wav_mel))

                img_batch, mel_batch = self.prepare_batch(img_batch, mel_batch, img_size)

                # pytorch 推理
                # start_model = time.time()
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                with torch.no_grad():
                    if a_flag != config_util.get_last_question() and not is_think:
                        print("跳出去！不在加入新的视频2")
                        return
                    if self.audio_smooth:
                        audio_embedding = self.model.audio_forward(mel_batch, a_alpha=1.25)
                        audio_embedding = 0.2 * audio_embedding[:-2] + 0.6 * audio_embedding[
                                                                             1:-1] + 0.2 * audio_embedding[2:]
                        pred = self.model.inference(audio_embedding, img_batch)
                    else:
                        pred = self.model(mel_batch, img_batch, a_alpha=1.25)

                gc.collect()
                pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.
                for p, f, c, af, inv_m in zip(pred, frames, coords, align_frames, inv_ms):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    af[y1:y2, x1:x2] = p
                    frames_list.append((af, f, inv_m))
                    if a_flag != config_util.get_last_question() and not is_think:
                        print("跳出去！不在加入新的视频3")
                        return
                    # 返回一个位置和标志位（标志是否需要进行传输）
                    pos, flag = self.check_handle(len(frames_list), self.gen_frame_num)
                    if flag:
                        # 单独开一个线程 这里需要线程池，单线程执行这里的逻辑，保证传输顺序
                        executor.submit(self.thread_write_frames, frames_list[pos:], content_text, play_mode,
                                        self.segment_data[0],a_flag,is_think)
                batch_data.clear()
        e = time.time()
        print('[info] 进度条时间：', e - start_infer)

        executor.shutdown(wait=True)
        end_infer = time.time()
        print('[info] 推理总耗时：', end_infer - start_infer)
        gc.collect()
        return self.frame_list_res

    def process_frames(self, num, frames_list, frame_queue):
        num = num * 1000000
        for af, f, inv_m in frames_list:
            frame = img_warp_back_inv_m(af, f, inv_m)
            frame_queue.put((num, frame))
            num += 1

    def thread_write_frames(self, frames_list, content_text, play_mode, segment_code,a_flag,is_think=False):
        thread_size = 1
        frames_len = len(frames_list)
        max_thread_size = self.build_frame_thread_size
        if frames_len < 200:
            thread_size = max_thread_size // 2
        else:
            thread_size = max_thread_size

        half_length = int((frames_len + thread_size - 1) / thread_size)  # 上取整
        g = time.time()
        frames = []
        for i in range(thread_size):
            frames_single = frames_list[i * half_length: (i + 1) * half_length]
            frames.append(frames_single)
        frames_list.clear()

        frame_queue = queue.Queue()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_size) as executor:
            futures = [executor.submit(self.process_frames, i + 1, frames[i], frame_queue) for i in range(thread_size)]
            # 等待所有线程完成
            for _ in as_completed(futures):
                pass
        print('[info] 生成帧总时长：', time.time() - g)
        g = time.time()
        frames_list = []
        while not frame_queue.empty():
            frames_list.append(frame_queue.get())
        frames_list.sort(key=lambda x: x[0])
        # cv2.resize(x, (1444, 3840))
        if self.avatar['frame_h'] == config_util.video_height:
            frames_list = [x for _, x in frames_list]
        else:
            frames_list = [cv2.resize(x, (config_util.video_width, config_util.video_height)) for _, x in frames_list]
        # frames_list = [x for _, x in frames_list]
        # resized_frames = []
        # for frame in frames_list:
        #     # 将帧上传到 GPU
        #     gpu_frame = cv2.cuda_GpuMat()
        #     gpu_frame.upload(frame)
        #     # 创建 GPU 上的 resize 对象
        #     resizer = cv2.cuda.createResize((1444, 3840))
        #     # 在 GPU 上进行 resize 操作
        #     resized_gpu_frame = resizer.resize(gpu_frame)
        #     # 下载结果到 CPU
        #     resized_frame = resized_gpu_frame.download()
        #     resized_frames.append(resized_frame)
        # frames_list = resized_frames
        if a_flag != config_util.get_last_question() and not is_think:
            print("跳出去！不在加入新的视频")
            return
        # print('[info] 生成帧：', self.wav_path.replace('.wav', '.mp3'),content_text)
        # height, width = frames_list[0][0].shape[:3]
        print(f"----->1Width: {frames_list[0][0].shape}")
        print('[info] 处理帧总时长：', time.time() - g)
        self.video_stream.offer(self.wav_path,
                                frames_list,
                                content_text,
                                play_mode,
                                self.gen_frame_num,
                                segment_code,a_flag=a_flag,is_think=is_think)
        self.frame_list_res.extend(frames_list)


class Wav2lip:

    # 初始化函数
    def __init__(self, checkpoint_path,
                 yolo_path,
                 facial_path,
                 data_path,
                 video_stream,
                 # face='resource/data/train.mp4',
                 face=None,
                 # pkl_data_path='resource/data/train.pkl',
                 pkl_data_path=None,
                 wav2lip_batch_size=16):
        self.checkpoint_path = checkpoint_path
        self.yolo_path = yolo_path
        self.facial_path = facial_path
        self.face = face
        self.pkl_data_path = pkl_data_path
        self.data_path = data_path
        self.wav2lip_batch_size = wav2lip_batch_size
        args = self.get_parser().parse_args()
        self.runner = Runner(args, self.yolo_path, self.facial_path, self.data_path, video_stream)

    # 包装参数
    def get_parser(self):
        parser = argparse.ArgumentParser(
            description='Inference code to lip-sync videos in the wild using Wav2Lip models')
        parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from',
                            default=self.checkpoint_path, required=False)
        parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use',
                            default=self.face, required=False)  # 原视频路径
        parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source',
                            default="", required=False)  # 音频路径
        parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                            default='')  # 合成视频保存路径
        parser.add_argument('--static', default=False, action='store_true',
                            help='If True, then use only first video frame for inference')
        parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                            default=15., required=False)
        parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0],
                            help='Padding (top, bottom, left, right). Please adjust to include chin at least')
        parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)',
                            default=self.wav2lip_batch_size)
        parser.add_argument('--resize_factor', default=1, type=int,
                            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

        parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                            help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                                 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

        parser.add_argument('--rotate', default=False, action='store_true',
                            help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                                 'Use if you get a flipped result, despite feeding a normal looking video')

        parser.add_argument('--pretrained_model_dir', type=str, default='weights', help='')

        parser.add_argument('--audio_smooth', default=True, action='store_true', help='smoothing audio embedding')

        parser.add_argument('--pkl_data_path', type=str, default=self.pkl_data_path,
                            help='数据预处理文件路径')  # 数据预处理文件路径
        return parser

    # 生成视频主函数
    # 输入： ----> 音频路径
    # 输出： ----> 视频路径
    def wav2lip(self, audio_path, text, mode,a_flag,is_think=False):
        return self.runner.run(audio_path, text, mode,a_flag,is_think)

    # 获取视频信息
    def get_video_info(self):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # frame_h, frame_w = self.runner.avatar['frame_h'], self.runner.avatar['frame_w']
        fps = self.runner.fps
        # print('fps:', fps)
        # print('frame_h:', frame_h)
        # print('frame_w:', frame_w)
        return fourcc, fps, (config_util.video_width,config_util.video_height)

print("------2>")
wav2lip_instance = None
def getWav2lip():
    global wav2lip_instance
    if wav2lip_instance is None:
        wav2lip_instance = Wav2lip(checkpoint_path='resource/checkpoints/yywav2lip.pth',
                               yolo_path='resource/yolov8n-face/yolov8n-face.pt',
                               facial_path='resource/wflw/hrnet18_256x256_p1/',
                               data_path='datas',
                               video_stream=config_util.video_stream,  # 传输video_stream
                               # face='resource/data/train.mp4',
                               face=config_util.video_stream.window.face,
                               # pkl_data_path='resource/data/train.pkl',
                               pkl_data_path=config_util.video_stream.window.pkl_data_path,
                               wav2lip_batch_size=config_util.batch_size)
    return wav2lip_instance

def chenge():
    global wav2lip_instance
    if wav2lip_instance is None:
        return
    wav2lip_instance.runner.avatar = None
    gc.collect()
    wav2lip_instance.runner.avatar = wav2lip_instance.runner.build_avatar(config_util.video_stream.window.pkl_data_path, config_util.video_stream.window.face, wav2lip_instance.runner.fps, wav2lip_instance.runner.args)
    print('Data preprocess loaded')


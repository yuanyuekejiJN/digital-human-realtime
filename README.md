<div align="center">
    <br>
    <h1>元岳数字人实时互动系统windows版</h1>
<p align="center">
  <img src="images/icon.png" alt="元岳logo" width="150" height="150">
</p>
</div>

## 项目简介

元岳数字人实时互动系统windows版是一个集成了语音识别、自然语言处理和语音合成等多种AI技术的开源解决方案，**具有精准对口型、多模型切换和流式输出等核心特点**。系统提供了灵活模块化的设计，支持多种大语言模型和语音服务，可以实现智能对话与虚拟形象的实时交互。基于先进的Wav2Lip技术，**系统能够实现语音与数字人唇形的精准同步**，同时支持**无限制添加数字人形象且多种数字人形象模型的无缝切换**，满足不同场景和用户需求。采用流式处理架构，系统可实现实时语音识别、处理和响应，提供流畅自然的交互体验，而模块化设计则使其易于扩展新功能和集成新的AI模型。通过本系统，开发者可以轻松创建适用于各种场景的智能数字助理，为用户提供个性化的语音和视觉交互体验。

### 数字人界面演示

以下是系统界面的实际效果展示：

#### 控制面板界面

![控制面板界面](/images/yuanyue.png)

#### 数字人交互界面

![数字人交互界面](/images/yuanyue7.png)

[[🔗 数字人界面演示视频（B站）](/images/yuanyue7.png)](https://www.bilibili.com/video/BV1hEe9zzETw/?vd_source=fb7533c84acaa1032d3a68827b2d1feb)

## 系统架构

## 功能列表和特点

| 功能模块 | 功能点     | 说明                 |
|---------|---------|--------------------|
| **智能对话** | 多模型支持   | Dify平台、GPT、ChatGLM2等多种大模型 |
| | 自然语言理解  | 语义分析与意图识别          |
| | 上下文记忆   | 支持多轮对话，记忆上下文信息     |
| **语音交互** | **实时语音识别**  | 阿里云、FunASR、科大讯飞多种引擎 |
| | 高质量语音合成 | 支持多种音色与情感表达        |
| | 情感语音    | 根据对话情绪自动调整语音风格     |
| | **唤醒词识别**   | **自定义唤醒词触发交互**     |
| **视觉表现** | 唇形同步    | **Wav2Lip技术实现精准唇形同步** |
| | 高清视频    | 支持高清视频输出           |
| **系统功能** | Web控制面板 | 基于Flask的管理界面       |
| | 数字人角色添加 | **可以无限制添加数字人角色形象** |
| | 多角色管理   | **多个数字人角色配置与切换**   |
| | 缓存系统    | 音频、视频、文本多级缓存       |
| | 通信协议    | 标准化WebSocket通信接口   |
| **扩展功能** | 多音色支持   | 阿里云多种音色(zhifeng_emo、zhimiao_emo等) |
| | 微软TTS   | 微软语音服务集成           |
| | 多模态交互   | 语音+文本双模式输入输出       |
| | 人设定制    | 可自定义角色设定与回答风格      |
| | 问答对管理   | 支持导入与编辑自定义问答对      |

### 核心组件

1. **主应用程序（`main.py`）**
   - 应用程序入口点
   - 授权验证管理
   - 初始化核心服务和UI

2. **核心模块**
   - `fay_core.py`：数字人核心功能
   - `wsa_server.py`：WebSocket服务器
   - `recorder.py`：音频录制
   - `tts_voice.py`：文本转语音管理
   - `interact.py`：交互消息处理
   - 数据库组件：`authorize_tb.py`、`content_db.py`、`shuziren_db.py`

3. **AI模块**
   - **语音识别(ASR)**：阿里云NLS、Funasr、科大讯飞
   - **文本转语音(TTS)**：阿里云TTS、微软TTS、Edge TTS
   - **自然语言处理**：支持多种大模型接入
     - GPT/ChatGPT
     - ChatGLM2
     - 灵聚
     - RWKV
     - Dify
   - **计算机视觉**：YOLOv8面部检测

4. **GUI组件**
   - 基于Flask的Web界面
   - PyQt5界面
   - 视频显示窗口

5. **通信接口**
   - WebSocket服务器(端口10002)：与数字人客户端通信
   - WebSocket服务器(端口10003)：与Web面板通信

## 安装与使用

### 系统与硬件要求

- Python 3.9、3.10
- 支持Windows
- 需要至少10G显存的英伟达显卡以及32G内存，对cpu也有一定要求

### 安装依赖

方法一：使用pip直接安装

```bash
pip install -r requirements.txt
```

方法二：使用Conda环境（推荐）

```bash
# 创建Python 3.10虚拟环境
conda create -n shuiziren python=3.10 -y

# 激活虚拟环境
conda activate shuiziren

# 安装依赖包
pip install -r requirements.txt
```
### 模型下载

```bash
# 可以看到到三个文件
链接: https://pan.baidu.com/s/18NndWwzc2o5x7VAAsG6VAQ 提取码: yykj 

# 其中yywav2lip.pth是用于音频驱动的唇形同步技术的模型。需要放置在目录 resource/checkpoints
# model_9.pkl和model_10.pkl是 是数字人面部数据预处理文件预处理的数字人面部关键点数据文件，
# 存储了视频帧中的面部关键点信息，用于实现音频驱动的唇形同步  需要分别放置在目录cache_data/9/model 和 cache_data/10/model
# yywav2lip.pth必须进行下载并放入对应的目录，model_9.pkl为默认的形象文件模型，同样必须下载并且放置在对应目录
```



### 配置说明

在使用前，需要在`system.conf`中配置相关API密钥和参数。以下是主要配置项的说明和示例（展示的key仅为示例）：

```
# ====================== 密钥配置 ======================
[key]
ASR_mode = ali

#   阿里云 实时语音识别 服务密钥（必须）https://ai.aliyun.com/nls/trans
ali_nls_key_id= TLAI5tFG9pKued5RGQofqx5t
ali_nls_key_secret= mHA8jDRBO7FP8pzRMUUwGVLqEsctM2
ali_nls_app_key= NaqZro6dcydNcGaH

# 语音合成模式选择: ali 
tts_mode = ali

# 阿里云文字转语音服务密钥
ali_tts_url = wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1
ali_tts_key_id=TLAI5tFG9pKued5RGQofqx5t
ali_tts_key_secret=mHA8jDRBO7FP8pzRMUUwGVLqEsctM2
ali_tts_app_key=NaqZro6dcydNcGaH


# NLP模型选择: lingju、yuan、gpt、chatgpt、rasa、chatglm2、VisualGLM、rwkv_api、rwkv、langcao、dify
chat_module= dify

# Dify服务配置
dify_url = http://yuanyuekj.com/v1/chat-messages
dify_key = app-xxx

# OpenAI GPT服务配置
chatgpt_api_key= sk-xxx
proxy_config=http://127.0.0.1:7890

# ====================== 系统配置 ======================
[conf]
# 初始状态是否静音
conf_muting=True

# MP4缓存模式: 1=numpy模式 2=numpy压缩模式 3=cv2视频写入存储
mp4_cache_mode=3

# 分段模式: 0=不分段 1=按帧数平均分段 2=按比例平均分段 3=按比例不平均分配
segment_mode=1
segment_basis=50

# 模型引擎配置（仅当chat_module选择为gpt时有效）
model_engine=gpt-3.5-turbo
```

**注意**：

1. 必须配置相应服务的API密钥才能正常使用各项功能
2. 详细配置说明请参考项目根目录下的`system.conf`文件
3. 不同模块的配置项可能需要配合使用，例如选择了ali模式的ASR/TTS，就需要配置对应的阿里云密钥
4. 问答对的设置在根目录下的qa_demo.xlsx进行编辑

### 启动系统

```bash
# 启动完整应用
python main.py
```


### 缓存管理

缓存数据存储在`cache_data`目录下，结构如下：

###

- `mp3`：重要MP3文件缓存
- `mp3_temp`：临时MP3文件缓存
- `mp4`：重要MP4文件缓存
- `mp4_temp`：临时MP4文件缓存
- `text_temp`：问答缓存

清除缓存示例：

```bash
# 清除MP4缓存
rm -rf cache_data/*/mp4/* cache_data/*/mp4_temp/*

# 清除所有缓存
rm -rf cache_data/*/mp3/* cache_data/*/mp3_temp/* cache_data/*/mp4/* cache_data/*/mp4_temp/* cache_data/*/text_temp/*
```

### 角色自定义

要更改和添加数字人角色：

1. 启动main.py
2. 在暴露的127.0.0.1:5000端口页面点击数字人管理,
3. 点击新增，输入名字和声音（声音目前支持阿里的男zhifeng_emo和女zhimiao_emo，如果想增加其他声音需要在阿里的tts查看支持的声音，参照<https://help.aliyun.com/zh/isi/developer-reference/overview-of-speech-synthesis?spm=a2c4g.11186623.help-menu-30413.d_3_1_0_0.73097a17SUfpT0&scm=20140722.H_84435._.OR_help-T_cn~zh-V_1>）
4. 上传一段无动作视频和一段有动作的视频即可添加角色成功。

### 通信协议

系统通过WebSocket与数字人客户端通信，消息格式包括：

1. **情绪信息**：情绪状态（-1到1）
2. **音频信息**：音频文件路径、持续时间、唇形同步数据
3. **文本信息**：用于显示响应文本
4. **问题信息**：用于显示用户问题
5. **日志信息**：用于系统日志

## 项目目录结构

```
.
├── main.py                # 程序主入口
├── fay_booter.py          # 核心启动模块
├── config.json            # 控制器配置文件
├── system.conf            # 系统配置文件
├── requirements.txt       # 依赖包列表
├── LICENSE                # GNU GPL v3 许可证
├── README.md              # 项目说明文档
├── WebSocket.md           # WebSocket协议说明
├── ai_module/             # AI模块目录
│   ├── ali_nls.py           # 阿里云语音识别
│   ├── ali_tts_sdk.py       # 阿里云语音合成
│   ├── funasr.py            # FunASR语音识别
│   ├── keda_nls.py          # 科大讯飞语音识别
│   ├── nlp_ChatGLM2.py       # ChatGLM2模型集成
│   ├── nlp_VisualGLM.py     # VisualGLM模型集成
│   ├── nlp_chatgpt.py       # ChatGPT集成
│   ├── nlp_dify.py          # Dify平台集成
│   ├── nlp_gpt.py           # GPT API集成
│   ├── nlp_langchao.py      # 浪潮模型集成
│   ├── nlp_lingju.py        # 灵聚平台集成
│   ├── nlp_rwkv.py          # RWKV模型集成
│   ├── nlp_yuan.py          # 浪潮源模型集成
│   ├── xf_ltp.py            # 讯飞情感分析
│   └── yolov8.py            # YOLOv8目标检测
├── core/                  # 核心模块目录
│   ├── fay_core.py          # 数字人核心功能
│   ├── wsa_server.py        # WebSocket服务器
│   ├── recorder.py          # 录音器模块
│   ├── tts_voice.py         # 语音合成管理
│   ├── interact.py          # 交互消息处理
│   ├── authorize_tb.py      # 授权表管理
│   ├── content_db.py        # 内容数据库
│   ├── qa_service.py        # 问答服务
│   ├── question_db.py       # 问题数据库
│   └── shuziren_db.py       # 数字人数据库
├── gui/                   # 图形界面目录
│   ├── flask_server.py      # Flask Web服务
│   ├── window.py            # 主窗口类
│   ├── video_window.py      # 视频窗口类
│   ├── video_window.ui      # 视频窗口UI设计
│   ├── video_window_3840.ui # 4K分辨率UI设计
│   ├── static/              # 静态资源文件
│   └── templates/           # HTML模板文件
├── scheduler/             # 调度器目录
│   └── thread_manager.py    # 线程管理器
├── utils/                 # 工具类目录
│   ├── config_util.py       # 配置工具
│   ├── storer.py            # 存储工具
│   └── util.py              # 通用工具类
├── resource/              # 资源目录
│   ├── checkpoints/         # 模型检查点
│   ├── wflw/                # 面部检测模型
│   └── yolov8n-face/        # YOLOv8人脸检测模型
├── wav2lip/               # 唇形合成模块
│   ├── inference_torch.py   # Wav2Lip推理引擎
│   ├── models/              # 模型目录
│   └── torchalign/          # 人脸对齐模块
├── cache_data/            # 缓存数据目录
    ├── mp3/                 # MP3重要文件
    ├── mp3_temp/            # 临时MP3缓存
    ├── mp4/                 # MP4重要文件
    ├── mp4_temp/            # 临时MP4缓存
    └── text_temp/           # 文本缓存



```

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

### 许可证与使用须知

- **开源使用**：遵循Apache License 2.0条款
- **定制与商务咨询**：如需更换页面logo或进行二次开发，请联系商务咨询
- **免责声明**：本软件按"原样"提供，不提供任何明示或暗示的保证
- **合规要求**：使用本软件时必须遵守相关法律法规，特别是AI内容安全和数据隐私法规

## 🙏 致谢

感谢以下优秀的开源项目：

[Fay 项目](https://github.com/xszyou/Fay)

[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)

## 💬 技术交流群

<div align="center">

<img src="images/企微交流群.png" alt="微信二维码" width="200" height="200"><br>
<strong>扫码进入该项目微信交流群</strong><br>
<em>邀请进群交流</em>

</div>


## 💼 商务合作

<div align="center">

<img src="images/LY-企微.png" alt="LY二维码" width="200" height="200"><br>
<strong>需要大模型/数字人/算法备案 商务咨询</strong><br>


<strong>商务合作咨询</strong><br>
</div>



---

<div align="center">
<p align="center">
  <strong>⭐ 如果本项目对你有帮助，请给个Star支持一下！</strong><br>
  <sub>🛡️ 专注AI内容安全，守护网络环境 | 由 yuanyueLLM AI 开源社区维护</sub>
</p>

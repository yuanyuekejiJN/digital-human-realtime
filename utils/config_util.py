import hashlib
import os
import json
import codecs
from collections import deque
from configparser import ConfigParser
from time import time

from exceptiongroup import catch
from sympy.strategies.core import switch

from wav2lip.utils import lazy_pinyin

config: json = None
system_config: ConfigParser = None
system_chrome_driver = None
key_ali_nls_key_id = None
key_ali_nls_key_secret = None
key_ali_nls_app_key = None
key_ms_tts_key = None
Key_ms_tts_region = None
key_xf_ltp_app_id = None
key_xf_ltp_api_key = None
key_ngrok_cc_id = None
key_yuan_1_0_account = None
key_yuan_1_0_phone = None
key_chatgpt_api_key = None
key_chat_module = None
key_gpt_access_token = None
key_gpt_conversation_id = None
proxy_config = None

ASR_mode = None
local_asr_ip = None
local_asr_port = None

tts_mode = None
key_ali_tts_url = None
key_ali_tts_key_id = None
key_ali_tts_key_secret = None
key_ali_tts_app_key = None


conf_muting = None
mute = False

# 0：
# 1：使用numpy模式存储
# 2：使用numpy压缩模式存储
# 3：使用cv2视频写入存储
mp4_cache_mode = None  # MP4缓存模式

app_config = None

video_stream = None    # 仅借用一下地方，与其他逻辑无关
last_question = ""
q_a_list = None
omega_key = ""
omega_url = ""
omega_model = ""
omega_emperature = ""
omega_max_tokens = ""
dify_url = ""
dify_key =""
lip_index = 0
frame = deque()
def set_last_question(text):
    global last_question
    last_question = text
    global lip_index
    lip_index = 0
    print("-->set_last_question->", last_question)

def set_last_question2(text):
    global last_question
    last_question = text
    print("-->set_last_question2->", last_question)

def get_last_question():
    global last_question
    # print("get_last_question-->", last_question)
    return last_question

face = None
pkl_data_path = None

segment_mode = None
segment_basis = None
feiFei = None
# 生成帧线程配置
build_frame_thread_size = None

batch_size = None

model_engine = None
model_prompt = None

# 语音、视频配置
default_voice = None
video_width = 1440
video_height = 2560


def load_config():
    global config
    global video_width
    global video_height
    global system_config
    global key_ali_nls_key_id
    global key_ali_nls_key_secret
    global key_ali_nls_app_key
    global key_ms_tts_key
    global key_ms_tts_region
    global key_xf_ltp_app_id
    global key_xf_ltp_api_key
    global key_ngrok_cc_id
    global key_yuan_1_0_account
    global key_yuan_1_0_phone
    global key_chatgpt_api_key
    global key_chat_module
    global key_gpt_access_token
    global key_gpt_conversation_id
    global key_lingju_api_key
    global key_lingju_api_authcode
    global proxy_config

    global ASR_mode
    global local_asr_ip
    global local_asr_port

    global tts_mode
    global key_ali_tts_url
    global key_ali_tts_app_key
    global key_ali_tts_key_secret
    global key_ali_tts_key_id

    global conf_muting
    global mp4_cache_mode

    # 分段配置
    global segment_mode
    global segment_basis

    # 生成帧线程配置
    global build_frame_thread_size
    global batch_size

    # GPT配置
    global model_engine
    global model_prompt

    # 语音、视频配置
    global default_voice

    global omega_key
    global omega_url
    global omega_model
    global omega_emperature
    global omega_max_tokens
    global dify_url
    global dify_key

    system_config = ConfigParser()
    system_config.read('system.conf', encoding='UTF-8')
    video_width = int(system_config.get('conf', 'width'))
    video_height = int(system_config.get('conf', 'height'))
    omega_key = system_config.get('key', 'omega_key')
    omega_url = system_config.get('key', 'omega_url')
    omega_model = system_config.get('key', 'omega_model')
    omega_emperature = system_config.get('key', 'omega_emperature')
    omega_max_tokens = system_config.get('key', 'omega_max_tokens')
    dify_url = system_config.get('key', 'dify_url')
    dify_key = system_config.get('key', 'dify_key')
    key_ali_nls_key_id = system_config.get('key', 'ali_nls_key_id')
    key_ali_nls_key_secret = system_config.get('key', 'ali_nls_key_secret')
    key_ali_nls_app_key = system_config.get('key', 'ali_nls_app_key')
    key_ms_tts_key = system_config.get('key', 'ms_tts_key')
    key_ms_tts_region  = system_config.get('key', 'ms_tts_region')
    key_xf_ltp_app_id = system_config.get('key', 'xf_ltp_app_id')
    key_xf_ltp_api_key = system_config.get('key', 'xf_ltp_api_key')
    key_ngrok_cc_id = system_config.get('key', 'ngrok_cc_id')
    key_yuan_1_0_account = system_config.get('key', 'yuan_1_0_account')
    key_yuan_1_0_phone = system_config.get('key', 'yuan_1_0_phone')
    key_chatgpt_api_key = system_config.get('key', 'chatgpt_api_key')
    key_chat_module = system_config.get('key', 'chat_module')
    key_gpt_access_token = system_config.get('key', 'gpt_access_token')
    key_gpt_conversation_id = system_config.get('key', 'gpt_conversation_id')
    key_lingju_api_key = system_config.get('key', 'lingju_api_key')
    key_lingju_api_authcode = system_config.get('key', 'lingju_api_authcode')

    ASR_mode = system_config.get('key', 'ASR_mode')
    local_asr_ip = system_config.get('key', 'local_asr_ip')
    local_asr_port = system_config.get('key', 'local_asr_port')

    proxy_config = system_config.get('key', 'proxy_config')

    tts_mode = system_config.get('key', 'tts_mode')
    key_ali_tts_url = system_config.get('key', 'ali_tts_url')
    key_ali_tts_key_id = system_config.get('key', 'ali_tts_key_id')
    key_ali_tts_key_secret = system_config.get('key', 'ali_tts_key_secret')
    key_ali_tts_app_key = system_config.get('key', 'ali_tts_app_key')

    conf_muting = system_config.get('conf', 'conf_muting')

    mp4_cache_mode = system_config.get('conf', 'mp4_cache_mode')

    # 分段配置
    segment_mode = int(system_config.get('conf', 'segment_mode'))
    segment_basis = system_config.get('conf', 'segment_basis')
    if segment_mode == 3:
        segment_basis = [float(item) for item in segment_basis.split(',')]
    elif segment_mode == 2:
        segment_basis = float(segment_basis)
    elif segment_mode == 1:
        segment_basis = int(segment_basis)

    # 生成帧线程配置
    build_frame_thread_size = int(system_config.get('conf', 'build_frame_thread_size'))
    batch_size = int(system_config.get('conf', 'batch_size'))

    # GPT配置
    model_engine = system_config.get('conf', 'model_engine')
    model_prompt = system_config.get('conf', 'model_prompt')


    # 语音、视频配置
    default_voice = system_config.get('conf', 'default_voice')


    config = json.load(codecs.open('config.json', encoding='utf-8'))
    if config['attribute']['huanxingchi']:
        config['attribute']['huanxingchi_r'] = ''.join(
            lazy_pinyin(config['attribute']['huanxingchi']))
    if config['attribute']['chufachi']:
        config['attribute']['chufachi_r'] = ''.join(
            lazy_pinyin(config['attribute']['chufachi']))
    if config['attribute']['cache_model_content']:
        config['attribute']['cache_model_content_r'] = config['attribute']['cache_model_content'].replace("，", ",").split(",")

def save_config(config_data):
    global config
    config = config_data
    if config['attribute']['huanxingchi']:
        config['attribute']['huanxingchi_r'] = ''.join(
            lazy_pinyin(config['attribute']['huanxingchi']))
    if config['attribute']['chufachi']:
        config['attribute']['chufachi_r'] = ''.join(
            lazy_pinyin(config['attribute']['chufachi']))
    if config['attribute']['cache_model_content']:
        config['attribute']['cache_model_content_r'] = config['attribute']['cache_model_content'].replace("，", ",").split(",")
    else:
        config['attribute']['cache_model_content_r'] = ""
    file = codecs.open('config.json', mode='w', encoding='utf-8')
    file.write(json.dumps(config, sort_keys=True, indent=4, separators=(',', ': ')))
    file.close()
    # for line in json.dumps(config, sort_keys=True, indent=4, separators=(',', ': ')).split("\n"):
    #     print(line)

def get_cache(question,answer = True):
    global config
    cache_content = config['attribute']['cache_model_content_r']
    print("config util +++++++++++++", config['attribute']["cache_model"])
    print(cache_content)
    if config['attribute']["cache_model"] == 0:
        return None
    elif config['attribute']["cache_model"] == 1:
        for string in cache_content:
            if string in question:
                return answer
        return None
    elif config['attribute']["cache_model"] == 2:
        for string in cache_content:
            if string in question:
                return None
        return answer

if __name__ == '__main__':
    config['attribute']['muting'] = False
    save_config(config)

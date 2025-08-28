import json

import markdown
import requests
import time
import threading
import fay_booter
import gui.video_window


from urllib3.exceptions import InsecureRequestWarning
from gui.video_window import VideoWindow
from utils import config_util as cfg, config_util

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

sentence_array = []
current_sentence = ""
def process_streaming_output(output_chunk):
    global current_sentence
    print('进入句子')
    current_sentence += output_chunk
    end_punctuation_marks = [".", "!", "?", "。", "！", "？"]
    if current_sentence[-1] in end_punctuation_marks:
        sentence_array.append(current_sentence[:-1])
        current_sentence = ""
    return sentence_array


def question(cont):
    url = cfg.dify_url
    # print("11111112222222222")
    response_text = ""
    buffer = ""

    # url = "http://localhost/v1/chat-messages"
    # url = "http://agentapi.yuanyuekj.com/v1/chat-messages"
    session = requests.Session()
    session.verify = False
    # print("2222222222222222222222")
    # video_window = VideoWindow()
    # print("333333333333333333333333")
    prompt = config_util.model_prompt
    # prompt = "你是数字人小O"
    msg = cont
    print("msgmsgsmgsmsmsm",msg)
    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": msg}
    ]
    # print(message)
    # data = {
    #     "messages": message,
    #     "temperature": 0.3,
    #     "max_tokens": 2000,
    #     "user": "live-virtual-digital-person"
    # }

    # data = {
    # "model": "gpt-3.5-turbo",
    # "stream": True,
    # "messages": message
    # }
    data = {
        "inputs": {},
        "query": msg,
        # "response_mode": "blocking",
        "response_mode": "streaming",
        "user": "abc-123"
    }

    # headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + cfg.omega_key}
    # headers = {'content-type': 'application/json', 'Authorization': 'Bearer '+ "hi1JMbYf8C8FFMCkMWvRmVHJaCzbK75b2CYNCUDkBq3miYQYXvl8Trc8M2RVCaLT"}
    headers = {'content-type': 'application/json', 'Authorization': 'Bearer '+ cfg.dify_key}
    starttime = time.time()
    # print("222223222222233")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            current_block = ''

            for line in response.iter_lines():
                if line:
                    # print(line)
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        # print(line.strip("data:").strip())
                        lines = line.strip("data:").strip()
                        data = json.loads(lines)
                        if data["event"] == "message" and data["answer"] != "":
                            answer = data["answer"]
                            buffer += answer
                            response_text += answer
                            if answer[-1] in ["!", "?", "。", "！", "？", ".\\n", "!\\n", "?\\n", "。\\n", "！\\n", "？\\n", "\\n",".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n", "！\\n\\n", "？\\n\\n", "\\n\\n"]:
                                print("--ans>", buffer, str(time.time() - starttime))
                                threading.Thread(target=fay_booter.speak, args=(msg, buffer)).start()
                                buffer = ""

        # if response.status_code == 200:
        #     current_block = ''
        #     for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
        #         current_block += chunk
        #
        #         if current_block.endswith('\n\n'):
        #             lines = current_block.strip("data:").strip()
        #             # print(lines)
        #             # print("11111111111111")
        #             # for line in lines:
        #             if lines.startswith('event: ping'):
        #                 continue
        #             else:
        #                 json1 = json.loads(lines)
        #                 if json1["event"] == "message" and json1["answer"] != "":
        #
        #                     answer = json1["answer"]
        #                     response_text += answer
        #                     buffer += answer
        #                     if answer[-1] in [".", "!", "?", "。", "！", "？", ".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n",
        #                                      "！\\n\\n", "？\\n\\n", "\\n\\n", "]"]:
        #                         print("--ans>", buffer, str(time.time() - starttime))
        #                         threading.Thread(target=fay_booter.speak, args=(msg, buffer)).start()
        #                         buffer = ""



                    current_block = ''





    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        e_text = "抱歉，我现在太忙了，休息一会，请稍后再试。"
        threading.Thread(target=fay_booter.speak, args=(msg, e_text)).start()
    except BaseException as e:
        print(f"请求失败2: {e}")
        e_text = "抱歉，我现在太忙了，休息一会，请稍后再试。"
        threading.Thread(target=fay_booter.speak, args=(msg, e_text)).start()
    print("接口调用耗时 :" + str(time.time() - starttime))
    print("response text:", response_text)
    return response_text
    # return result
    # return list1

import re
if __name__ == "__main__":
    # query = "介绍一下济南"
    # thread = threading.Thread(target=question, args=(query,))
    # thread.start()
    # # 主线程可以继续执行其他任务
    # time.sleep(10)  # 让主线程等待一段时间
    # thread.join()  # 等待线程结束


    chinese_characters = []
    for i in range(1):
        query = "介绍一下济南5000字"
        response = question(query)
        print(response)

        # for chunk in response.iter_content(chunk_size=1024):
        #     if chunk:
        #         # 将字节数据解码为字符串
        #         text = chunk.decode('utf-8')
        #         # 使用正则表达式提取汉字
        #         chinese_matches = re.findall(r'[\u4e00-\u9fff]', text)
        #         chinese_characters.extend(chinese_matches)
        #         print(chinese_characters)
        # else:
        #     print(f"请求失败，状态码: {response.status_code}")
        # print(response)
        # buffer = ""
        # for choice in response.get("choices", []):
        #     delta = choice.get("delta", {})
        #     content = delta.get("content", "")
        #     print(content)
        # for line in response:
        #     if line:
        #         # 将字节数据解码为字符串
        #         # line = line.decode('utf-8')
        #         buffer += line
        #         # 检查 buffer 是否包含一个完整的句子
        #         sentences = buffer.split('. ')
        #         if len(sentences) > 1:
        #             for i in range(len(sentences) - 1):
        #                 print(sentences[i] + '。')
        #             buffer = sentences[-1]
        # print("\n The result is ", response)

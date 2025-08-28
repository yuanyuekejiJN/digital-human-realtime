
import json

import requests
import time
import threading
import fay_booter



from urllib3.exceptions import InsecureRequestWarning
from gui.video_window import VideoWindow
from utils import config_util

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
    # url = cfg.omega_url
    # print("11111112222222222")
    response_text = ""
    buffer = ""
    url = "https://api.openai-sb.com/v1/chat/completions"
    session = requests.Session()
    session.verify = False
    # print("2222222222222222222222")
    # video_window = VideoWindow()
    # print("333333333333333333333333")
    prompt = config_util.model_prompt
    # prompt = "你是数字人小O"
    msg = cont
    print(msg)
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

    data = {
    "model": "gpt-3.5-turbo",
    "stream": True,
    "messages": message
    }

    # headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + cfg.omega_key}
    # headers = {'content-type': 'application/json', 'Authorization': 'Bearer '+ "hi1JMbYf8C8FFMCkMWvRmVHJaCzbK75b2CYNCUDkBq3miYQYXvl8Trc8M2RVCaLT"}
    headers = {'content-type': 'application/json', 'Authorization': 'Bearer '+ "sb-8bec1c1e735848a90c9c4b7c5d980d21d719041a301a85c0"}
    starttime = time.time()
    print("222223222222233")
    try:
        response = session.post(url, json=data, headers=headers, verify=False,stream=True)
        # print("11111111111111")
        # response.raise_for_status()  # 检查响应状态码是否为200
        # json_data = response.text.replace("data: ", "")
        # print(json_data)
        # json_data = response.text.replace("data:[DONE]", "")
        # if json_data != "[DONE]":
        # result = json.loads(json_data)
        # print(result)

        # result = response.text
        # print("response iter lines", response.iter_lines())

        # while True:
        # print(response.iter_lines())

        for line in response.iter_lines():
            if msg != config_util.get_last_question():
                return ""
            if line:
                line = line.decode('utf-8')
                print("--line>", line,str(time.time() - starttime))
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = line.strip("data:").strip()
                    json1 = json.loads(data)
                    if "choices" in json1 and "delta" in json1["choices"][0]:
                        delta = json1["choices"][0]["delta"]
                        content = delta.get("content", "")
                        buffer += content
                        response_text += content
                        # print(buffer)
                        # 检查 buffer 是否包含一个完整的句子
                        # print("--content>", content, str(time.time() - starttime))
                        if content in ["!", "?", "。", "！", "？", ".\\n", "!\\n", "?\\n", "。\\n", "！\\n", "？\\n", "\\n",".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n", "！\\n\\n", "？\\n\\n", "\\n\\n"]:
                            print("--ans>", buffer,str(time.time() - starttime))
                            threading.Thread(target=fay_booter.speak, args=(msg, buffer)).start()
                            # print(buffer)
                            # yield buffer
                            # list1.append(buffer)
                            buffer = ""

                        # sentences = buffer.split('。')
                        # if len(sentences) > 1:
                        #     for i in range(len(sentences) - 2):
                        #         # print(sentences[i])
                        #         answer = sentences[i]
                        #         print("--ans>",answer)
                        #         threading.Thread(target=fay_booter.speak, args=(msg, answer)).start()
                        #     buffer = ""


                # print("111111111111111")
                # # 将字节数据解码为字符串
                # while 1:
                #     if len(line) == 12:
                #         continue
                #     else:
                #         line = line.decode('utf-8')
                #         # print(line)
                #         json_data = line.replace("data: ", "")
                #         json1 = json.loads(json_data)
                #         # print("json1:", json1)
                #         # print(len(json1["choices"][0]["delta"]))
                #         # print(len(json1),json1)
                #         if len(json1["choices"][0]["delta"]) == 0 :
                #             # print(json1["choices"][0]["delta"])
                #             # print("长度为0")
                #             continue
                #         else:
                #             # print(json1["choices"][0]["delta"])
                #             choice = json1["choices"][0]["delta"]["content"]
                #         # delta = choice.get("delta", {})
                #         # content = delta.get("content", "")
                #         buffer += choice
                #         # print(buffer)
                #         # 检查 buffer 是否包含一个完整的句子
                #         # print("11111111111111")
                #         sentences = buffer.split('。')
                #         if len(sentences) > 1:
                #             # buffer = sentences[-1]
                #             for i in range(len(sentences) - 1):
                #                 # print(sentences[i])
                #                 answer = sentences[i]
                #                 print("--ans>",answer)
                #                 threading.Thread(target=fay_booter.speak, args=(msg, answer)).start()
                #                 # fay_booter.speak(q_msg=msg, text=answer)
                #                 # yield sentences[i] + '。'
                #             buffer = sentences[-1]


                            #     # print(sentences[i] + '。')
                            #     return sentences[i] + "。"
                        # buffer = sentences[-1]
        # print(result)
        # sentence = process_streaming_output(result)
        # print("sentence:", sentence)


        # response_text = result["choices"][0]["delta"]["content"]


    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        e_text = "抱歉，我现在太忙了，休息一会，请稍后再试。"
        threading.Thread(target=fay_booter.speak, args=(msg, e_text)).start()
    except BaseException as e:
        print(f"请求失败2: {e}")
        e_text = "抱歉，我现在太忙了，休息一会，请稍后再试。"
        threading.Thread(target=fay_booter.speak, args=(msg, e_text)).start()
    print("接口调用耗时 :" + str(time.time() - starttime))

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
        # query = "请用python代码写一个排序算法"
        query = "介绍一下济南"
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

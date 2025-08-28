
import json
import threading

import requests
import time

from urllib3.exceptions import InsecureRequestWarning

import fay_booter
from utils import config_util as cfg, config_util

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)




def question(cont):
    response_text = ""
    buffer = ""
    url = cfg.omega_url
    # url = "https://10.110.63.144:36667/v1/chat/completions"
    session = requests.Session()
    session.verify = False

    prompt = cfg.model_prompt
    # prompt = "你是数字人小O"
    msg = cont

    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": msg}
    ]

    data = {
        "model": cfg.omega_model,
        "messages": message,
        "temperature": cfg.omega_emperature,
        "max_tokens": cfg.omega_max_tokens,
        "stream": True,
        # "user": "live-virtual-digital-person"
    }

    headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + cfg.omega_key}
    # headers = {'content-type': 'application/json', 'Authorization': 'Bearer '+ "hi1JMbYf8C8FFMCkMWvRmVHJaCzbK75b2CYNCUDkBq3miYQYXvl8Trc8M2RVCaLT"}
    starttime = time.time()

    try:
        # response = session.post(url, json=data, headers=headers, verify=False)
        # response.raise_for_status()  # 检查响应状态码是否为200
        #
        # result = json.loads(response.text)
        # response_text = result["choices"][0]["message"]["content"]
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
                # print("--line>", line, str(time.time() - starttime))
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
                        if content in [".", "!", "?", "。", "！", "？",".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n", "！\\n\\n", "？\\n\\n","\\n\\n"]:
                            print("--ans>", buffer, str(time.time() - starttime))
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


if __name__ == "__main__":

    for i in range(3):
        query = "介绍一下济南"
        response = question(query)
        print("\n The result is ", response)

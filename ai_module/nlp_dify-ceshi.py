import requests
import json
import time
import markdown

# 替换为实际的 API 密钥
# api_key = "app-K82g8yj0ACSK2kAUE8OiGXRI"
#
# # 请求的 URL
# url = 'http://agentapi.yuanyuekj.com/v1/workflows/run'


api_key = "app-BPdndUj1asRaighDvhVvpoKb"
url = 'http://localhost/v1/chat-messages'

# 请求头
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# print("1111")
# 请求体数据
data = {
    "inputs": {},
    "query":"用python写一个排序算法",
    # "content":"你好",
    # "response_mode": "streaming",
    "response_mode": "blocking",
    "user": "abc-123"
}

print("1111")
buffer = ""
text = ""
starttime = time.time()
try:
    # response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    # for line in response.iter_lines():
    #     # print("line:", line)
    #     # if msg != config_util.get_last_question():
    #     #     return ""
    #     if line:
    #
    #         line = line.decode('utf-8')
    #         # print("line:", line)
    #         # print("--line>", line,str(time.time() - starttime))
    #         if line.startswith("data:") and not line.startswith("data: [DONE]"):
    #             data = line.strip("data:").strip()
    #             json1 = json.loads(data)
    #             print(json1)
    #             if "choices" in json1 and "delta" in json1["choices"][0]:
    #                 delta = json1["choices"][0]["delta"]
    #                 content = delta.get("content", "")
    #                 buffer += content
                    # response_text += content
                    # print(buffer)
                    # 检查 buffer 是否包含一个完整的句子
                    # print("--content>", content, str(time.time() - starttime))
                    # if content in [".", "!", "?", "。", "！", "？", ".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n", "！\\n\\n",
                    #                "？\\n\\n", "\\n\\n"]:
                    #     print("--ans>", buffer, str(time.time() - starttime))
                        # threading.Thread(target=fay_booter.speak, args=(msg, buffer)).start()
                        # print(buffer)
                        # yield buffer
                        # list1.append(buffer)
                        # buffer = ""





    #
    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    response.encoding = 'utf-8'
    print(response.status_code)
    print(response.text)
    data = json.loads(response.text)
    answer = data['answer']
    print(answer)


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
                        text += answer
                        # print(answer[-1])
                        if answer[-1] in ["!", "?", "。", "！", "？", ".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n",
                                                                      "！\\n\\n", "？\\n\\n", "\\n\\n"]:
                            print("--ans>", buffer, str(time.time() - starttime))
                            buffer = ""
        # print(markdown.markdown(text))





            # current_block += chunk
            # print("2222")
            # print(chunk)
            # prefix = b'data: '
            # if chunk.startswith(prefix):
            #     byte_data = chunk[len(prefix):]
            #     print(byte_data)
            # if chunk.startswith("b'data:"):
            #     lines = chunk.strip("data:").strip()
            #     print("原始信息",lines)
            #     if lines.startswith('event: ping'):
            #         continue
            #     else:
            #         # print(lines)
            #         json1 = json.loads(lines)
            #         # print(json1["event"])
            #         print(json1)
            #         if json1["event"] == "message" and json1["answer"] != "":
            #             # print(json1)
            #
            #         # if "choices" in json1 and "delta" in json1["choices"][0]:
            #             delta = json1["answer"]
            #             # print(delta)
            #             # print(delta[-1])
            #             # content = delta.get("content", "")
            #             buffer += delta
            #             if delta[-1] in [".", "!", "?", "。", "！", "？", ".\\n\\n", "!\\n\\n", "?\\n\\n", "。\\n\\n",
            #                              "！\\n\\n", "？\\n\\n", "\\n\\n", "]"]:
            #                 print("--ans>", buffer, str(time.time() - starttime))
            #                 # threading.Thread(target=fay_booter.speak, args=(msg, buffer)).start()
            #                 # print(buffer)
            #                 # yield buffer
            #                 # list1.append(buffer)
            #                 buffer = ""
            #
            #         current_block = ''


    # print("12222")
    # # 检查响应状态码
    # # print(response.text)
    # response.raise_for_status()
    # # print(response.raise_for_status())  # 检查响应状态码是否为200
    # data = json.loads(response.text)
    # print(data)
    # # response_text = data["data"]["outputs"]["out"]
    # response_text = data["answer"]
    # print(response_text)
    print(time.time() - starttime)
except requests.exceptions.HTTPError as http_err:
    print(f'HTTP 错误发生: {http_err}')
except requests.exceptions.RequestException as req_err:
    print(f'请求发生错误: {req_err}')
except json.JSONDecodeError as json_err:
    print(f'JSON 解析错误: {json_err}')


import json

def is_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

# 测试示例
valid_json = {
    "inputs": {},
    "query":"用python写一个排序算法",
    "response_mode": "blocking",
    "user": "abc-123"
}
invalid_json = '{"name": "John", age: 30}'  # 注意这里的键 "age" 没有引号，是无效的 JSON

print(is_json(valid_json))  # 输出: True
print(is_json(invalid_json))  # 输出: False
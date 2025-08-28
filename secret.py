import base64
import binascii
import datetime
import json

import wmi
from Crypto.Cipher import AES


def aes_encrypt(key, data):
    aes = AES.new(key, AES.MODE_ECB)
    padded_data = data + (16 - len(data) % 16) * chr(16 - len(data) % 16)
    encrypted_data = aes.encrypt(padded_data.encode('utf-8'))
    encrypted_text = base64.b64encode(encrypted_data).decode('utf-8')
    return encrypted_text

def aes_decrypt(key, encrypted_data):
    aes = AES.new(key, AES.MODE_ECB)
    decrypted_data = aes.decrypt(base64.b64decode(encrypted_data))
    padding_len = decrypted_data[-1]
    return decrypted_data[0: - padding_len].decode('utf-8')
import subprocess

def get_windows_product_id():
    try:
        s = wmi.WMI()
        mainboard = []
        for board_id in s.Win32_BaseBoard():
            mainboard.append(board_id.SerialNumber.strip().strip('.'))
        return mainboard[0]

    except Exception as e:
        print("获取 Windows 产品 ID 出错:", e)
        return None

if __name__ == '__main__':
    try:
        print(get_windows_product_id())
        file_path = "secret.bin"  # 定义保存的文件路径和文件名，可以根据实际情况修改
        key = b'abcdefghijklmnop'  # AES - 128密钥长度为16字节
        given_date = datetime.datetime(2025, 12, 5, 14, 0, 0)
        # 将datetime对象转换为时间戳（浮点数形式），再转换为整数
        timestamp_seconds = int(given_date.timestamp())
        # print(timestamp_seconds)
        info = {
            "id": "BFEBFBFF00090672",
            "time":timestamp_seconds
        }
        data = json.dumps(info)
        print(data)
        encrypted_result = aes_encrypt(key, data)
        print("加密结果:", encrypted_result)
        # print("加密结果:", encrypted_result[0: - padding_len].decode('utf- 8'))
        decrypted_result = aes_decrypt(key, encrypted_result)
        print("解密结果:", decrypted_result)
        # with open(file_path, 'w') as file:
        #     file.write(encrypted_result)
    except BaseException as e:
        print(e)
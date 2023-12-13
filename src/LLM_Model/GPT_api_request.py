import http.client
import json

def send_request(message):
    conn = http.client.HTTPSConnection("api.aikey.one")
    payload = json.dumps({
       "model": "gpt-3.5-turbo",
       "messages": [
          {
             "role": "user",
             "content": message
          }
       ]
    })
    headers = {
       'Authorization': 'sk-zQFauHVV8S9k6kzo4829929b83A44229B37cE723921f3c26',  # 请用您的实际 API 密钥替换 'key'
       'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
       'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data1 = res.read()

    # 解析 JSON 字符串
    response_data = json.loads(data1.decode("utf-8"))

    # 提取并返回 content
    # 检查 'choices' 是否存在且不为空
    if "choices" in response_data and len(response_data["choices"]) > 0:
        answer = response_data["choices"][0]["message"]["content"]
        print(answer)






#sk-zQFauHVV8S9k6kzo4829929b83A44229B37cE723921f3c26
import json
import requests

apiKey = "ef46baec5e4c75cbcbabc3b6b241ff4e"

def addrressToCoord(addr):
    url = f'https://dapi.kakao.com/v2/local/search/address.json?query={addr}'
    headers = {"Authorization": "KakaoAK " + apiKey}
    result = json.loads(str(requests.get(url, headers=headers).text))
    if len(result['documents']) != 0:
        matchFirst = result['documents'][0]['address']
        return float(matchFirst['x']), float(matchFirst['y'])
    else:
        return None

print(addrressToCoord('제주특별자치도 서귀포시 안덕면 신화역사로1097번길 178'))

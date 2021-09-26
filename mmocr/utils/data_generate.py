import os
import json
import requests
from base64 import b64encode
import cv2


def data_json(image_path, data_set, image_name):
    data_json={}
    with open(image_path, 'rb') as jpg_file:
       byte_content = jpg_file.read()
    #print((byte_content))
    # byte_content = cv2.imread(image_path)
    base64_bytes = b64encode(byte_content)
    #data_json["img"] = base64_bytes #byte_content
    data_json["img"] = base64_bytes.decode('utf-8')
    data_json["data_set"] = data_set
    data_json["index"] = [image_name, '1']
    
    data_json['img'] = [data_json['img'], data_json['img']] #.tolist()
    data = json.dumps(data_json).encode()
    return data

def pre_process(data):
    print("my_pre_process.")
    data = data.get_data()
    #json process
    json_data = json.loads(data.decode('utf-8'))
    image_base64_string = json_data.get("img")
    image_data = base64.b64decode(image_base64_string)
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_name = json_data.get("index")
    return (img,img_name)


if __name__ == "__main__":
    print("="*10)
    img_dir = "/home/will/tianchi_ocr/train"
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        data = data_json(img_path, "test1", img_file)
        requests.post(url="http://127.0.0.1:8080/tccapi", data=data)
        #pre_process(data)


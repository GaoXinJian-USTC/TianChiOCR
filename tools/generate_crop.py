import os, sys, json
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw
import os
import json
import numpy as np
import pandas as pd
from urllib.parse import unquote
import random
from tqdm import tqdm
import shutil

path = '/home/will/gaoxinjian/ocr_data/train/'

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)

    offset = int(n_total * ratio)

    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    val = full_list[:offset]
    train = full_list[offset:]
    return val,train


class Rotate(object):
    def __init__(self, image, coordinate, angle=0):
        self.image = image.convert('RGB')
        self.coordinate = coordinate
        self.xy = [tuple(self.coordinate[k]) for k in ['left_top', 'right_top', 'right_bottom', 'left_bottom']]
        self._mask = None
        self.image.putalpha(self.mask)
        self.angle = angle

    @property
    def mask(self):
        if not self._mask:
            mask = Image.new('L', self.image.size, 0)
            draw = ImageDraw.Draw(mask, 'L')
            draw.polygon(self.xy, fill=255)
            self._mask = mask
        return self._mask

    def run(self):
        image = self.rotation_angle()
        box = image.getbbox()
        return image.crop(box)

    def rotation_angle(self):
        x1, y1 = self.xy[0]
        x2, y2 = self.xy[1]
        # angle = self.angle([x1, y1, x2, y2], [0, 0, 10, 0]) * -1
        # print(self.angle)
        return self.image.rotate(self.angle, expand=True)


train = [
"/home/will/dataset/csv/Xeon1OCR_round1_train_20210524.csv",
"/home/will/dataset/csv/Xeon1OCR_round1_train1_20210526.csv",
"/home/will/dataset/csv/Xeon1OCR_round1_train2_20210526.csv",
]

valset_num = 10000
train_rec_df = []
angle_dict = {"底部朝下":0, "底部朝右":270, "底部朝上":180, "底部朝左":90}

random_seed = [i for i in range(230000)]

val_list,train_list = data_split(random_seed,0.06,True)

for csv in train:
    train_rec_df.append(pd.read_csv(csv,encoding='utf-8'))
train_rec_df = pd.concat(train_rec_df)

idx = 0

# shutil.rmtree("/home/will/gaoxinjian/ocr_data")
os.makedirs("/home/will/gaoxinjian/ocr_data/det_images/")

train_label = open('/home/will/gaoxinjian/ocr_data/train_list.txt', 'a+',encoding='utf-8')
val_label = open('/home/will/gaoxinjian/ocr_data/val_list.txt',"a+",encoding='utf-8')

total_num = len(train_rec_df)

for row in tqdm(train_rec_df.iloc[:].iterrows(), total=total_num):
    path = unquote(json.loads(row[1]['原始数据'])['tfspath'])
    img_path = "/home/will/dataset/train/" + unquote(path.split('/')[-1])
    # print(img_path)
    labels = json.loads(row[1]['融合答案'])[0]
    orientation = json.loads(row[1]['融合答案'])[1].get("option", "底部朝下")

    # try:
    image = Image.open(img_path)
    for label in labels:
        name = str(idx).zfill(7)
        text = json.loads(label['text'])['text']
        text.replace(" ","")
        if text == "" or text == None:
            continue
        coord = [int(float(x)) for x in label['coord']]
        coordinate = {'left_top': coord[:2], 'right_top': coord[2:4], 'right_bottom': coord[4:6], 'left_bottom': coord[-2:]}
        direction = json.loads(label['text']).get("direction","底部朝下")
        angle = angle_dict[orientation] + angle_dict[direction]
        rotate_ = Rotate(image, coordinate, angle)
        corp_img = rotate_.run().convert('RGB')
        corp_img.save(f'/home/will/gaoxinjian/ocr_data/det_images/train_{name}.jpg')
        if not idx in val_list:
            train_label.write(f'det_images/train_{name}.jpg {text}\n')
        else:
            val_label.write(f'det_images/train_{name}.jpg {text}\n')
        idx += 1
    # except Exception as e:
    #     print("error",e)

for c in ["train","val"]:
    eval(f"{c}_label.close()")

# Copyright (c) OpenMMLab. All rights reserved.
import copy
from mmocr.utils import ocr
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
import cv2
import traceback
import sys
import mmcv
import numpy as np
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config

from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm
from ai_hub import inferServer
import json
import base64


class MMOCR():

    def __init__(self,
                 det='DRRG',
                 det_config='/mmocr/configs/textdet/drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                 det_ckpt='/mmocr/models/42525.pth',
                 recog='SAR',
                 recog_config='/mmocr/configs/textrecog/sar/sar_r31_parallel_decoder_chinese.py',
                 recog_ckpt='/mmocr/models/epoch_13.pth',
                 kie='',
                 kie_config='',
                 kie_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device='cuda:0',
                 **kwargs):

        textdet_models = {
            'DB_r18': {
                'config':
                'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20210325-91cef9af.pth'
            },
            'DRRG': {
                'config': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth'
            },
            'FCE_IC15': {
                'config': 'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt': 'fcenet/fcenet_r50_fpn_1500e_icdar2015-d435c061.pth'
            },
            'FCE_CTW_DCNv2': {
                'config': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500-05d740bb.pth'
            },
            'MaskRCNN_CTW': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config':
                'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
            }
        }

        textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt': 'nrtr/nrtr_r31_academic_20210406-954db95e.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20210406-ce16e7cc.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20210809-59c8c92d.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20210811-2badf6fc.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            }
        }

        kie_models = {
            'SDMGR': {
                'config': 'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
                'ckpt':
                'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
            }
        }

        self.td = det
        self.tr = recog
        self.kie = kie
        self.device = device

        # Check if the det/recog model choice is valid
        if self.td and self.td not in textdet_models:
            raise ValueError(self.td,
                             'is not a supported text detection algorthm')
        elif self.tr and self.tr not in textrecog_models:
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')
        elif self.kie:
            if self.kie not in kie_models:
                raise ValueError(
                    self.kie, 'is not a supported key information extraction'
                    ' algorithm')
            elif not (self.td and self.tr):
                raise NotImplementedError(
                    self.kie, 'has to run together'
                    ' with text detection and recognition algorithms.')

        self.detect_model = None
        if self.td:
            # Build detection model
            if not det_config:
                det_config = os.path.join(config_dir, 'textdet/',
                                          textdet_models[self.td]['config'])
                
            if not det_ckpt:
                det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/' + \
                    textdet_models[self.td]['ckpt']

            self.detect_model = init_detector(
                det_config, det_ckpt, device=self.device)
            
            self.detect_model = revert_sync_batchnorm(self.detect_model)

        self.recog_model = None
        if self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = os.path.join(
                    config_dir, 'textrecog/',
                    textrecog_models[self.tr]['config'])
            if not recog_ckpt:
                recog_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'textrecog/' + textrecog_models[self.tr]['ckpt']

            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
            self.recog_model = revert_sync_batchnorm(self.recog_model)

        self.kie_model = None
        if self.kie:
            # Build key information extraction model
            if not kie_config:
                kie_config = os.path.join(config_dir, 'kie/',
                                          kie_models[self.kie]['config'])
            if not kie_ckpt:
                kie_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'kie/' + kie_models[self.kie]['ckpt']

            kie_cfg = Config.fromfile(kie_config)
            self.kie_model = build_detector(
                kie_cfg.model, test_cfg=kie_cfg.get('test_cfg'))
            self.kie_model = revert_sync_batchnorm(self.kie_model)
            self.kie_model.cfg = kie_cfg
            load_checkpoint(self.kie_model, kie_ckpt, map_location=self.device)

        # Attribute check
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module
            if model.cfg.data.test['type'] == 'ConcatDataset':
                model.cfg.data.test.pipeline = \
                    model.cfg.data.test['datasets'][0].pipeline


    # End2end ocr inference pipeline
    def inference(self, data):
        # Find bounding boxes in the images (text detection)
        # det_result = self.single_inference(det_model, self.args.arrays,
        #                                    self.args.batch_mode,
        #                                    self.args.det_batch_size)
        try:
            img_np = data[0]
            img_name = data[1]
            
            res = model_inference(self.detect_model,data[0], batch_mode=False)

            score_thr = 0.6
            valid_boundary_res = [
            res for res in res['boundary_result'] if res[-1] > score_thr
        ]
            bboxes = valid_boundary_res
            points = []
            transcriptions = []
            resdict ={}
            for bbox in bboxes:
                box = bbox[:8]
                if len(bbox) > 9:
                    min_x = min(bbox[0:-1:2])
                    min_y = min(bbox[1:-1:2])
                    max_x = max(bbox[0:-1:2])
                    max_y = max(bbox[1:-1:2])
                    box = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                box = [int(c) for c in box]
                points.append(box)
                box_img = crop_img(img_np, box)
                recog_result = model_inference(self.recog_model, box_img)
                transcriptions.append(recog_result['text'])
                
            resdict[img_name] = {
            "pointsList": points,
            "transcriptionsList": transcriptions,
            "ignoreList": [False] * len(points),
            "classesList": [1] * len(points)
            }
        except Exception as e:
                exc_type, exc_value, exc_obj = sys.exc_info()
                traceback.print_tb('error is :{} \n {}'.format(e,exc_obj))
                resdict = {}
                resdict["name"] = {
                    "pointsList": [[196, 2228, 2731, 2228, 2731, 2357, 196, 2357]],
                    "transcriptionsList": [["transcriptions"]],
                    "ignoreList": [False] * 1,
                    "classesList": [1] * 1
                    }
                return json.dumps(resdict)
            
        return json.dumps(resdict)



class myserver(inferServer):
    def __init__(self,model):
        super().__init__(model)
        # print("init_myserver")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = device
        # self.model= model.to(device)

    def pre_process(self, data):
        # print("my_pre_process.")
        data = data.get_data()
        #json process
        json_data = json.loads(data.decode('utf-8'))
        image_base64_string = json_data.get("img")[0]
        image_data = base64.b64decode(image_base64_string)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img_name = json_data.get("index")
        if isinstance (img_name, list):
            img_name = img_name[0]
        return (img,img_name)


    #pridict default run as follow：
    def predict(self, data):
        
        # moc data for test, usr should use pred data and key 
        
        resdict = self.model.inference(data)
        

    def post_process(self, data):
        return data


if __name__ == '__main__':
    # mymodel = mymodel()
    # myserver = myserver(mymodel)
    #run your server, defult ip=localhost port=8080 debuge=false
    model = MMOCR()
    myserver = myserver(model)
    myserver.run("127.0.0.1",8080,debuge=True) #myserver.run("127.0.0.1", 1234)

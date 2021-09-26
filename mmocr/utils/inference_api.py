from ai_hub import inferServer
import json
import random
import base64
import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
import json


class myserver(inferServer):
    def __init__(self,model):
        super().__init__(model)
        print("init_myserver")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = device
        # self.model= model.to(device)

    def pre_process(self, data):
        print("my_pre_process.")
        data = data.get_data()
        #json process
        json_data = json.loads(data.decode('utf-8'))
        image_base64_string = json_data.get("img")[0]
        image_data = base64.b64decode(image_base64_string)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img_name = json_data.get("index")
        return (img,img_name)

    #pridict default run as follow：
    def predict(self, data):
       
        ret = self.model(data[0])
        # moc data for test, usr should use pred data and key 
        key = data[1]
        points = []
        transcriptions = []
        resdict ={}
        for line in ret:
            points.append(sum(line[0], []))
            transcriptions.append(line[1][0])
            resdict[key] = {
                "pointsList": points,
                "transcriptionsList": transcriptions,
                "ignoreList": [False] * len(points),
                "classesList": [1] * len(points)
            }
        return json.dumps(resdict)

    def post_process(self, data):
        return data

# class mymodel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(1, 1)
#         self.model = lambda x: torch.mul(x , 2)

#     def forward(self, x):
#         y = self.model(x)
#         # y = self.fc(y)
#         return y


if __name__ == '__main__':
    # mymodel = mymodel()
    # myserver = myserver(mymodel)
    #run your server, defult ip=localhost port=8080 debuge=false
    mymodel = PaddleOCR(gpu_mem=8000,
                # det_model_dir="ch_ppocr_server_v2.0_det_infer",
                det_model_dir="models/ch_ppocr_server_v2.0_det_infer",
                cls_model_dir="models/ch_ppocr_mobile_v2.0_cls_infer",
                rec_model_dir="models/ch_ppocr_server_v2.0_rec_infer",
                use_angle_cls=True,
                cls=True
                )
    myserver = myserver(mymodel)
    myserver.run("127.0.0.1",8080,debuge=True) #myserver.run("127.0.0.1", 1234)
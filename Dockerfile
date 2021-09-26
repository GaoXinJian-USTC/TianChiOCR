FROM mmocr:full

RUN mkdir /mmocr

ADD . /mmocr/

WORKDIR /mmocr

RUN pip install ai_hub flask

RUN pip install -r requirements.txt

RUN pip install mmocr

RUN rm -rf SAR SATRN 

RUN rm -rf /root/.cache/pip 




import cv2
import gradio as gr
import onnxruntime
from face_judgement_align import IDphotos_create
from hivisionai.hycv.vision import add_background
from layoutCreate import generate_layout_photo, generate_layout_image
import pathlib
import numpy as np
from flask import Flask,request,Response
import base64
import json

RED_ID=1
BLUE_ID=2
WHITE_ID=3
RED_RGB=(233, 51, 35)
BLUE_RGB=(86, 140, 212)
WHITE_RGB=(255, 255, 255)
bg_color_dict={RED_ID:RED_RGB,BLUE_ID:BLUE_RGB,WHITE_ID:WHITE_RGB}



app=Flask(__name__)

@app.route('/',methods=["POST"])
def root():
    img_base64=base64.b64decode(request.json["img"].split(';base64,')[1]) 
    img_buff = np.frombuffer(img_base64, dtype='uint8')
    image = cv2.imdecode(img_buff,1)
    if request.json["size"]==1:
        size=(413, 295)
    elif request.json["size"]==2:
        size=(626, 413)
    else:
        result={"succeed":False,"note":"尺寸错误或数据损坏，请重试"}
        return Response(json.dumps(result),  mimetype='application/json')
    
    #try:
    result_image_hd, result_image_standard, typography_arr, typography_rotate, \
         _, _, _, _, status = IDphotos_create(image,
                                         size=size,
                                         align=False,
                                         beauty=False,
                                         fd68=None,
                                         human_sess=sess,
                                         IS_DEBUG=False)
    '''
    except Exception:
        print()
        result={"succeed":False,"note":"发生异常,请重试"}
        return Response(json.dumps(result),  mimetype='application/json')
        '''
    result_image_hd = np.uint8(add_background(result_image_hd, bgr=bg_color_dict[request.json["bg_color"]]))
    typography_arr, typography_rotate = generate_layout_photo(size[0], size[1])
    six_inch_layout=generate_layout_image(result_image_hd, typography_arr,
                                                        typography_rotate,
                                                        height=size[0],
                                                        width=size[1])
    success, encoded_image = cv2.imencode(".jpg", result_image_hd)
    if success!=True:
        result={"succeed":False,"note":"base64编码错误,请重试"}
        return Response(json.dumps(result),  mimetype='application/json')
    success, encoded_image_layout = cv2.imencode(".jpg", six_inch_layout)
    if success!=True:
        result={"succeed":False,"note":"base64编码错误,请重试"}
        return Response(json.dumps(result),  mimetype='application/json')
    
    byte_data = encoded_image.tobytes()
    base_str=base64.b64encode(byte_data).decode('utf-8')
    num=4-len(base_str)%4
    for i in range(num):
        base_str+='='
    base = "data:image/jpg;base64," + base_str

    byte_data_layout=encoded_image_layout.tobytes()
    base_layout_str= base64.b64encode(byte_data_layout).decode('utf-8')
    num=4-len(base_layout_str)%4
    for i in range(num):
        base_layout_str+='='
    base_layout="data:image/jpg;base64," + base_layout_str

    result={"success":True,"note":"成功","img":base,"img_layout":base_layout}
    return Response(json.dumps(result),  mimetype='application/json')

     
     


if __name__ == "__main__":
    HY_HUMAN_MATTING_WEIGHTS_PATH = "./hivision_modnet.onnx"
    sess = onnxruntime.InferenceSession(HY_HUMAN_MATTING_WEIGHTS_PATH,providers='CPUExecutionProvider')
    app.run()
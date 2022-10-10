from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from openvino.runtime import Core
import base64
import urllib.request as ur


def model_init(model):
    ie_core = Core()
    model = ie_core.read_model(model=model)
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model



with open("Live.txt", 'r') as f:
    str= f.read()



def detect_emotion(image_encoded):
    prototxt = "deploy.prototxt.txt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    ir_model_xml = "emotion.xml"
    ir_model_bin = "emotion.bin"
    inputs, outputs, compiled_model = model_init(ir_model_xml)

    class_labels = []
    file = "labels_emotion.txt"
    f = open(file, "r").readlines()
    for x in f:
        txt = x.split("\n")[0]
        class_labels.append(txt)

    image_decoded = ur.urlopen(image_encoded)
    img = image_decoded.file.read()
    np_data = np.frombuffer(img, np.uint8)
    cap = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    while True:
        width = min(1500, cap.shape[1])
        cap = cv2.resize(cap, (width, width))
        (h, w) = cap.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(cap, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = cap[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            boxes = compiled_model([face])[outputs]
            for i in range(5):
                format = "{:.2f}".format(boxes[0][i]*100)
                with open("results.txt", 'a') as outfile:
                    outfile.write(class_labels[i]+"="+format+"%" +'\n') 
                    
            pred = np.argmax(boxes)
            label = class_labels[pred]
            return(label)

print(detect_emotion(str))

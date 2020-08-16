from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from  tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import platform
import argparse




print("\n\nClose Window with ESC shortcut key\n\n")

model = load_model("model/cnn_mask.h5")

arg = argparse.ArgumentParser()
arg.add_argument("-s","--save",type=str,action="store",help="path of save frame")
arg.add_argument("-c","--confidence",type=float,required=True,nargs="?",const=0.5,help="Confidence point of face detection(default = 0.5)")
args = arg.parse_args()

data_class_face = [1]
data_array_face = np.vstack(data_class_face)
data_class_mask = [0]
data_array_mask = np.vstack(data_class_mask)
os_name = ["test"]
fit_save = 0 

load_model_caffe = cv2.dnn.readNetFromCaffe("caffemodel/deploy.prototxt",'caffemodel/weights.caffemodel')
check_os = platform.system()
if args.save:
    fit_save +=1
    if check_os == "Linux":
        os_name.append("XVID")
    elif check_os == "Windows":
        os_name.append("DIVX")

def image_prepare(img):
    roi_resize_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_resize_color)
    img_array = img_to_array(pil_img)
    img_array = np.expand_dims(img_array,axis=0)
    img_array = img_array/255
    return  img_array

def face_detection(img,width,height):
    resize_img = cv2.resize(img,(300,300))
    search_blob = cv2.dnn.blobFromImage(resize_img,1.0, (300, 300), (104.0, 177.0, 123.0))
    load_model_caffe.setInput(search_blob)
    detection = load_model_caffe.forward()
    for i in range(0,detection.shape[2]):
        bounding_box = detection[0,0,i,3:7]*np.array([width,height,width,height])
        (x,y,w1,h1) = bounding_box.astype('int')

        confidence = detection[0,0,i,2]
        if confidence > 0.5:
            return  (x,y,w1,h1)

cap = cv2.VideoCapture(0)
#frame_per_second = cap.get(cv2.CAP_PROP_FPS)
widths = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
heights = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
save = cv2.VideoWriter(args.save,cv2.VideoWriter_fourcc(*os_name[fit_save]),20,(widths,heights))
while True:
    ret, frame = cap.read()
    frame_copy = frame.copy()
    find_detecting = face_detection(frame_copy, widths, heights)
    try:
        (x, y, w, h) = find_detecting
        roi = frame_copy[y:h, x:w]
        roi_resize = cv2.resize(roi, (150, 150))
        image_generator = image_prepare(roi_resize)
        cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 1)
        result = model.predict_classes(image_generator)
        if result == data_array_face:
            cv2.putText(frame, "without mask", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        elif result == data_array_mask:
            cv2.putText(frame, "with mask", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    except:
        pass
    if args.save:
        save.write(frame)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
save.release()
cap.release()
cv2.destroyAllWindows()

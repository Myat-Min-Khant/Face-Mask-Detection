from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from  tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import platform
import time
from PIL import Image
from imutils.video import FPS

print("\nClose Window With ESC shortcut key\n\n")
arg = argparse.ArgumentParser()
arg.add_argument("-i","--input",type=str,action="store",help="path of input image")
arg.add_argument("-s","--save",type=str,action="store",help="path of save image or video")
arg.add_argument("-v","--video",type=str,action='store',help="path of input video")
arg.add_argument("-c","--confidence",type=float,required=True,nargs='?', const=0.5,help="Confidence point of face detection(default = 0.5)") 
args = arg.parse_args()

model = load_model("model/cnn_mask.h5")
data_class_face = [1]
data_array_face = np.vstack(data_class_face)
data_class_mask = [0]
data_array_mask = np.vstack(data_class_mask)
check_os = platform.system()

# load  Caffe model for detecting face
load_model_caffe = cv2.dnn.readNetFromCaffe("caffemodel/deploy.prototxt",'caffemodel/weights.caffemodel')

def image_prepare(img):
    roi_resize = cv2.resize(img,(150,150))
    roi_resize = cv2.cvtColor(roi_resize,cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_resize)
    img_array = img_to_array(pil_img)
    img_array = np.expand_dims(img_array,axis=0)
    img_array = img_array/255
    return  img_array




def image(path):
    read_image = cv2.imread(path)
    (h,w) = read_image.shape[:2]
    resize_img = cv2.resize(read_image, (300, 300))
    search_blob = cv2.dnn.blobFromImage(resize_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    load_model_caffe.setInput(search_blob)
    detection = load_model_caffe.forward()
    for i in range(0, detection.shape[2]):
        bounding_box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, w1, h1) = bounding_box.astype('int')

        confidence = detection[0, 0, i, 2]
        if confidence > args.confidence:
            roi = read_image[y:h1,x:w1]
            img_generator = image_prepare(roi)
            predict_mask = model.predict_classes(img_generator)
            if predict_mask == data_array_face:
                cv2.rectangle(read_image, (x, y), (w1, h1), (0, 0, 200), 4)
                cv2.putText(read_image, "", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,0), 1)
            elif predict_mask == data_array_mask:
                cv2.rectangle(read_image,(x,y),(w1,h1),(255,255,0),4)
                cv2.putText(read_image, "mask", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            pass
        

    while True:
        cv2.imshow("image",read_image)
        if cv2.waitKey(1) & 0xFF == 27:
            if args.save:
                cv2.imwrite(args.save,read_image)
            break
    cv2.destroyAllWindows()

def video(path): 
    os_name = ['test']
    fit_save = 0

    if args.save:
        fit_save += 1
        if check_os == "Linux":
            os_name.append("XVID")
        elif check_os == "Windows":
            os_name.append("DIVX")

    cap = cv2.VideoCapture(path)
    fps = FPS().start()
    frame_per_second = cap.get(cv2.CAP_PROP_FPS)
    widths = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heights = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*os_name[fit_save]),frame_per_second,(widths,heights))
    if cap.isOpened() == False:
        print("Wrong path of video file or something is wrong!")
    while cap.isOpened():
        ret, frame = cap.read()
        (h,w) = frame.shape[:2]
        if ret == True:
            frame_copy = frame.copy()
            resize_img = cv2.resize(frame_copy, (300, 300))
            search_blob = cv2.dnn.blobFromImage(resize_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
            load_model_caffe.setInput(search_blob)
            detection = load_model_caffe.forward()
            for i in range(0, detection.shape[2]):
                bounding_box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, w1, h1) = bounding_box.astype('int')
                confidence = detection[0, 0, i, 2]
                try:

                    if confidence > args.confidence:
                        roi = frame[y:h1,x:w1]
                        img_generator = image_prepare(roi)
                        result = model.predict_classes(img_generator)
                        if result == data_array_face:
                            cv2.rectangle(frame, (x, y), (w1, h1), (0, 0, 200), 4)
                            cv2.putText(frame, "", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200),1)
                        elif result == data_array_mask:
                            cv2.rectangle(frame, (x, y), (w1, h1), (255, 255, 0), 4)
                            cv2.putText(frame, "with mask", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                    else:
                        pass
                except:
                    pass


            
            
            if args.save:
                save.write(frame)
            	
            time.sleep(1/frame_per_second)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break


            fps.update()
        
        else:
            break
    save.release()
    fps.stop()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__' :
    if args.input:
        image(args.input)
    elif args.video:
        video(args.video)










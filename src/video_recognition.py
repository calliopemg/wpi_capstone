import cv2
import os
import numpy as np
from time import sleep


def facial_recognition(in_file, out_file):
    # Load a cascade file for detecting faces
    haarcascades_path = '../opencv/data/haarcascades/'
    face_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye.xml')

    # reading the input
    cap = cv2.VideoCapture(f"{in_file}")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # use 10 instead for slow motion

    output = cv2.VideoWriter(f"{out_file}", fourcc, fps, (frame_width, frame_height), True)

    while True:
        ret, frame = cap.read()
        if ret:
            faces = face_cascade.detectMultiScale(frame)

            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (fx + 6, fy - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

                roi_color = frame[fy:fy + fh, fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(roi_color)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, 'Eye', (ex + 6, ey - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            # write the new frame
            output.write(frame)
        else:  # EOF
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
    print("Finished facial_recognition on", in_file)


def animal_recognition(in_file, out_file):
    # define the directory and name for model
    onnx_model_path = "../opencv/data/matt_model/"
    onnx_model_name = "best.onnx"

    # get full path to the converted model
    full_model_path = os.path.join(onnx_model_path, onnx_model_name)

    # read converted .onnx model with OpenCV API
    opencv_net = cv2.dnn.readNetFromONNX(full_model_path)
    print("OpenCV model was successfully read. Layer IDs: \n", opencv_net.getLayerNames())

    # reading the input
    cap = cv2.VideoCapture(f"{in_file}")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # use 10 instead for slow motion

    output = cv2.VideoWriter(f"{out_file}", fourcc, fps, (frame_width, frame_height), True)

    while True:
        ret, frame = cap.read()
        if ret:
            # ------- Code for inference part ----------
            blob = cv2.dnn.blobFromImage(frame, 1, (224, 224))
            opencv_net.setInput(blob)
            out = opencv_net.forward()
            # Let us grab the top5 probability indices
            idx_asc = out.argsort()
            idx_desc = np.fliplr(idx_asc)
            idx_top5 = idx_desc[0, :5]

            r = 1
            # Now for each of the index create a text and put on displayed frame
            for id in idx_top5:
                #text = "{}: probability {:.3} %".format(classes[id], out[0, id] * 100)  # FIXME
                text = "{}: probability {:.3} %".format("ANIMAL", out[0, id] * 100)
                cv2.putText(frame, text, (0, 25 + 40 * r), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                r += 1

            # write the new frame
            output.write(frame)
        else:  # EOF
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
    print("Finished animal_recognition on", in_file)

# run on-demand to perform object recognition on saved motion clips
if __name__ == '__main__':
    # assign directory
    src_dir = 'raw'
    out_dir = 'output'

    # iterate over files in the source directory
    print("Starting animal_recognition on files in", src_dir)
    for filename in os.scandir(src_dir):
        if filename.is_file():
            print(filename.path)
            #facial_recognition(filename.path, str(filename.path + "_out.mp4"))
            animal_recognition(filename, str(filename.path + "_out.mp4"))

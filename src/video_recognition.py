import cv2
import os
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
    # Load a cascade file for detecting animals
    haarcascades_path = '../opencv/data/haarcascades/'
    # TODO: add animal classifier
    animal_cascade = cv2.CascadeClassifier(haarcascades_path + 'tbd.xml')

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
            animals = animal_cascade.detectMultiScale(frame)

            for (fx, fy, fw, fh) in animals:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                cv2.putText(frame, 'Animal', (fx + 6, fy - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

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
    print("Starting facial_recognition on files in", src_dir)
    for filename in os.scandir(src_dir):
        if filename.is_file():
            print(filename.path)
            facial_recognition(filename.path, str(filename.path + "_out.mp4"))
            #animal_recognition(filename, filename + "_out.mp4")

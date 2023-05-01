import cv2
from time import sleep
from picamera2 import MappedArray, Picamera2, Preview
from libcamera import Transform


def facial_recognition(image):
    # Load a cascade file for detecting faces
    haarcascades_path = '../opencv/data/haarcascades/'
    face_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Look for faces in the image using the loaded cascade file
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    print("Found " + str(len(faces)) + " face(s)")

    # Draw a rectangle around every found face
    count = 0
    for (fx, fy, fw, fh) in faces:
        # write the face as a separate file
        cv2.imwrite('face_' + str(count) + '.jpg', image[fy:fy + fh, fx:fx + fw])
        count += 1

        cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = image[fy:fy + fh, fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Save the result image
    cv2.imwrite('faces_rectangles.jpg', image)


def animal_recognition(image):
    # Load a cascade file for detecting animals
    haarcascades_path = '../opencv/data/haarcascades/'
    # TODO: add animal classifier
    animal_cascade = cv2.CascadeClassifier(haarcascades_path + 'tbd.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Look for animals in the image using the loaded cascade file
    animals = animal_cascade.detectMultiScale(gray, 1.1, 5)
    print("Found " + str(len(animals)) + " animal(s)")

    # Draw a rectangle around every found animal
    count = 0
    for (fx, fy, fw, fh) in animals:
        # write the face as a separate file
        cv2.imwrite('animal_' + str(count) + '.jpg', image[fy:fy + fh, fx:fx + fw])
        count += 1

        cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

    # Save the result image
    cv2.imwrite('animals_rectangles.jpg', image)


def camera():
    picam2 = Picamera2()

    # Get the picture
    # Here you can also specify other parameters (e.g.:rotate the image)
    # preview_config = picam2.create_preview_configuration(main={"size": (1024, 768), "format": "RGB888"})
    still_config = picam2.create_still_configuration(transform=Transform(vflip=1))
    picam2.configure(still_config)

    picam2.start()
    sleep(1)  # do we need to sleep here? Giving the camera a chance to "auto-adjust"?
    image = picam2.capture_array()

    # Save the original image
    cv2.imwrite('original.jpg', image)

    return image


if __name__ == '__main__':
    # TODO: only save image if detectMultiScale returns objects (Object Detection)
    # TODO: take picture every x seconds instead of once per script run
    image = camera()
    facial_recognition(image)
    # animal_recognition(image)

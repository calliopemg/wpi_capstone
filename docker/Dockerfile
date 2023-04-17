FROM debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
        && apt-get upgrade -y \
        && apt-get install -y -qq wget gnupg \
        && wget -qO- https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | apt-key add - \
        && echo "deb http://archive.raspberrypi.org/debian/ bullseye main" > /etc/apt/sources.list.d/raspi.list \
        && apt-get update \
        && apt-get install -y libcamera-apps-lite python3-picamera2 python3-opencv ffmpeg \
        && apt-get clean

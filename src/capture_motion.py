from datetime import datetime

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
from libcamera import Transform, controls


# detect motion and output a file
def detect_motion(vflip=False):
    lsize = (640, 480)
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"},
                                                     lores={"size": lsize, "format": "YUV420"},
                                                     transform=Transform(vflip=vflip))
    picam2.configure(video_config)
    encoder = H264Encoder()
    picam2.encoder = encoder

    # lower framerate
    #picam2.set_controls({"FrameRate": 10})

    # continuously autofocus (supported by Camera Module 3)
    #picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

    picam2.start()

    # run autofocus cycle once
    if picam2.autofocus_cycle():
        print("INFO: Autofocus Cycle successful")
    else:
        print("ERROR: Autofocus Cycle unsuccessful")

    w, h = lsize
    prev = None
    encoding = False

    while True:
        cur = picam2.capture_buffer("lores")
        cur = cur[:w * h].reshape(h, w)

        if prev is not None:
            # Measure pixel differences between current and previous frame (Mean Square Error)
            mse = np.square(np.subtract(cur, prev)).mean()
            if mse > 8:
                now = datetime.now()
                if not encoding:
                    encoder.output = FfmpegOutput(f"{int(now.timestamp())}.mp4", audio=False)

                    picam2.start_encoder(quality=Quality.VERY_HIGH)
                    encoding = True

                    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")  # YYYY-mm-dddd H:M:S
                    print("INFO: New Motion Detected at", dt_string, " MSE:", mse)
            else:
                if encoding and datetime.now().timestamp() - now.timestamp() > 5.0:
                    picam2.stop_encoder()
                    encoding = False
        prev = cur
    picam2.stop()


if __name__ == '__main__':
    detect_motion(vflip=True)

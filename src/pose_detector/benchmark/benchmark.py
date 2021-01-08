import json

import cv2
import requests


def run(input_img_path, num_requests=1000):
    """Performs a benchmark.

    Benchmark using by repeatedly sending an image over http to a model running
    on the docker container tensorflow/serving.

    Args:
        input_img_path: The image to use for the benchmark.
        num_requests: How many requests to send.
    """

    data = _create_payload(input_img_path)

    total_time = 0

    # The first few requests may take longer
    _warmup(data)

    print("Benchmark started")

    for i in range(num_requests):
        response = _send(data)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()

        percent = (100. * i / num_requests)
        print('\r%3d%%' % percent, end='', flush=True)

    print("\r100% Complete")
    print("avg latency: {} ms".format((total_time * 1000) / num_requests))


def _create_payload(path):
    """Reads the image and creates a json payload.
    """

    input_string = cv2.imread(str(path), 1).astype('float32').tolist()
    data = {"inputs": [input_string]}
    data = json.dumps(data)
    return data


def _warmup(data):
    for i in range(5):
        _send(data)


def _send(data):
    """Sends the data to the default tensorflow/serving port.
    """

    response = requests.post("http://localhost:8501/v1/models/pose_detection:predict", data=data,
                             headers={"content-type": "application/json"})
    return response

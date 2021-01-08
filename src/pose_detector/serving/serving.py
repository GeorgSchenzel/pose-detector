import docker


def run(model_path):
    """Runs the tensorflow/serving container.

    Args:
        model_path: The directory where the model to be served is stored.
    """

    client = docker.from_env()
    container = client.containers.run("tensorflow/serving:latest-gpu",
                                      runtime="nvidia",
                                      ports={'8501/tcp': 8501},
                                      volumes={str(model_path): {'bind': "/models/pose_detection", 'mode': 'rw'}},
                                      environment=["MODEL_NAME=pose_detection"],
                                      detach=True)

    try:
        for line in container.logs(stream=True):
            print(line.strip())

    except KeyboardInterrupt:
        print("Stopping container")
        container.stop()

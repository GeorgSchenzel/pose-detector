from pose_detector.generation import processing
from pose_detector.generation.rendering import Renderer


def generate_dataset(size, config_path, models_path, backgrounds_path, output_path, parallel=1, mode="all"):
    """Generates a dataset.

    Generates a dataset by rendering images using Blenderproc and then processing them
    further. Images are all saved to a single directory.

    The label for each image must be specified in the config file. It is encoded in the
    output name of each file, using the following format: "<img_num>_<label>.png"

    Args:
        size: The target size of the dataset to generate.
        models_path: The path to a directory containing .blend files with the models that should be rendered.
        config_path: The path to the configuration file for Blenderproc. It must follow the
          the config format specified in the Blenderproc documentation. It is also required
          to contain some specific modules. It is recommended to use the provided template
          for this file.
        output_path: Directory where the dataset will be stored.
        backgrounds_path: Directory where the backgrounds are stored.
        parallel: How many processes to start in parallel.
        mode: What action to perform; Options:
          "all": Render and process
          "render": Only perform the rendering step
          "process": Only perform the processing step
    """

    # Blenderproc will change the working directory so we need to resolve these paths
    # to ensure it is absolute.
    models_path = models_path.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    backgrounds_path = backgrounds_path.resolve()

    if mode == "all" or mode == "render":
        renderer = Renderer(count=size,
                            config_path=config_path,
                            model_paths=list(models_path.glob("*.blend")),
                            output_path=output_path,
                            parallel=parallel)
        renderer.render()

    if mode == "all" or mode == "process":
        processing.process_images(backgrounds=list(backgrounds_path.glob("*.jpg")),
                           output_path=output_path,
                           delete_tmp=-True)

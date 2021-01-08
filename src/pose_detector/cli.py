import argparse
from pathlib import Path

import pose_detector.generation.generator as generator
import pose_detector.training.training as training
import pose_detector.benchmark.benchmark as benchmark
import pose_detector.serving.serving as serving
import pose_detector.utility.utility as utility


def main():
    """The entry point of pose-detector
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    datagen_parser = subparsers.add_parser("generate",
                                           help="Create a dataset by rendering images using blender and applying "
                                                "post processing",
                                           description="""Create a dataset by rendering images using blender and 
                                           applying post processing.
                                           
                                           The Blenderproc pipeline is used to simplify and speed up the rendering 
                                           process in Blender. A config file for this tool must be provided. This 
                                           tool needs some specific Blenderproc modules to achieve the correct output 
                                           for further processing. It is highly recommended to use the provided 
                                           config template file `resources/template.yaml`. 

                                            By adding modules to the config you can modify the rendered images in any 
                                            way you want.""")
    datagen_parser.add_argument("config_path", type=Path, metavar="config",
                                help="Path to the configuration file for BlenderProc")
    datagen_parser.add_argument("models_path", type=Path, metavar="models",
                                help="Path to the directory containing the models as individual .blend files")
    datagen_parser.add_argument("backgrounds_path", type=Path, metavar="backgrounds",
                                help="Path of the directory containing background images")
    datagen_parser.add_argument("output_path", type=Path, metavar="output",
                                help="Path to the directory where the images should be stored")
    datagen_parser.add_argument("--mode", type=str, default="all", choices=["all", "render", "process"],
                                help="Only perform one step of the generation")
    datagen_parser.add_argument("--size", "-s", type=int, default=10,
                                help="The target size of the dataset to be generated, this might vary by a few images "
                                     "depending on the number of models and parallelization")
    datagen_parser.add_argument("--parallel", "-p", type=int, default=1,
                                help="How many process to use in parallel for rending the images")
    datagen_parser.set_defaults(func=generator.generate_dataset)

    train_parser = subparsers.add_parser("train",
                                         help="Train a CNN using a previously created dataset.",
                                         description="""Train a CNN using a previously created dataset.
                                                                                  
                                         This uses transfer learning on the `resnet18` model pretrained on the 
                                         `imagenet` dataset. It tries to predict the value encoded in the image name 
                                         created by the generation step. It will train for 20 epochs and then save 
                                         the generated model.
                                         """)
    train_parser.add_argument("images_directory", type=Path, metavar="dataset",
                              help="Path to the directory where the dataset is stored")
    train_parser.add_argument("save_path", type=Path, metavar="output",
                              help="Path where the trained model should be saved to")
    train_parser.set_defaults(func=training.run)

    benchmark_parser = subparsers.add_parser("benchmark",
                                             help="Perform a benchmark using a model that is being served with the "
                                                  "'serve' command",
                                             description="Perform a benchmark using a model that is being served with "
                                                         "the 'serve' command")
    benchmark_parser.add_argument("input_img_path", type=Path, metavar="image",
                                  help="Path of an image to use for benchmarking, must be 128x128 pixels")
    benchmark_parser.set_defaults(func=benchmark.run)

    serve_parser = subparsers.add_parser("serve",
                                         help="Serve a saved model using the tensorflow/serving docker container.",
                                         description="Serve a saved model using the tensorflow/serving docker container.")
    serve_parser.add_argument("model_path", type=Path, metavar="model",
                              help="Path to the saved model to be served, must be absolute")
    serve_parser.set_defaults(func=serving.run)

    visualize_parser = subparsers.add_parser("visualize",
                                             help="Create a collage of images including the labels.",
                                             description="Create a collage of images including the labels.")
    visualize_parser.add_argument("images_directory", type=Path, metavar="images",
                                  help="Path to the directory where the images are stored")
    visualize_parser.add_argument("save_path", type=Path, metavar="output",
                                  help="Path where the image should be saved to")
    visualize_parser.set_defaults(func=utility.visualize_dataset)

    args = parser.parse_args()

    if "func" not in args:
        parser.print_help()
    else:
        args_dict = vars(args)
        func = args_dict.pop("func")
        func(**args_dict)


if __name__ == "__main__":
    main()

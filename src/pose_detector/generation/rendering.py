import pathlib
from concurrent.futures.process import ProcessPoolExecutor
from math import ceil
from timeit import default_timer as timer
import subprocess

MAX_PER_PROCESS = 2000


class Renderer:
    """Used for rendering images using blenderproc.

    Allows to run generation concurrently.

    Attributes:
        count: How large the generated dataset will be.
        config_path: The path to the configuration file for Blenderproc..
        model_paths: The paths to .blend files containing the models that should be rendered..
        output_path: Directory where the images will be stored.
        parallel: How many processes to start in parallel.
        blender_path: Location of the directory containing the blender executable.
        blenderproc_run_path: Location of the blenderproc python script used as the entry point.
    """

    def __init__(self, count, config_path, model_paths, output_path, parallel=1):
        self.count = count
        self.config_path = config_path
        self.model_paths = model_paths
        self.output_path = output_path
        self.parallel = parallel

        root = _get_root_path()
        self.blender_path = root.joinpath("tools/blender")
        self.blenderproc_run_path = root.joinpath("tools/BlenderProc/run.py")

    def render(self):
        """Starts the rendering procedure.

        Splits up work into multiple chunks to reduce the Blenderproc overhead by limiting the maximum
        number of images a process might render.
        """

        print("Starting rendering")
        start = timer()

        run_index = 0

        # Blenderproc has overhead that scales with the number of images rendered
        # to avoid too long rendering times we cap the amount of rendered images
        # per process
        per_model = self.count // len(self.model_paths)
        iterations = ceil(per_model / MAX_PER_PROCESS)
        iterations = max(iterations, self.parallel // len(self.model_paths))  # force small datasets to parallelize
        per_iteration = per_model // iterations

        with ProcessPoolExecutor(self.parallel) as executor:
            for i in range(iterations):
                for model_path in self.model_paths:
                    executor.submit(self._start_render_process, run_index, str(model_path), per_iteration)
                    run_index += 1

        end = timer()
        print("Rendering completed in {}s".format(end - start))

    def _start_render_process(self, run_index, model_path, per_iteration):
        """Starts a single rendering process.

        Args:
            run_index: The index of this chunk of data that is being generated.
            model_path: The model to use for rendering this chunk.
            per_iteration: How many images should be rendered.
        """

        print("Starting run {}".format(run_index))

        tmp_dir = self.output_path.joinpath("tmp_{}_with_{}".format(run_index, per_iteration))
        p = subprocess.Popen(
            ["python", self.blenderproc_run_path,
             self.config_path,       # config
             self.blender_path,      # blender install location
             tmp_dir,                # temporary output directory
             model_path]             # .blend model to use
            + [str(per_iteration)],  # count to render
            stdout=subprocess.DEVNULL
        )
        p.wait()


def _get_root_path():
    """

    Returns: The path of the package root.
    """

    return pathlib.Path(__file__).joinpath("../../../../").resolve()

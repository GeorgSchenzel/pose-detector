from pathlib import Path
from mathutils import Vector

import bpy

from src.main.Module import Module


class CameraOffset(Module):
    """
    Centers the camera to an object's origin. The centering is done by offsetting the camera each frame after
    the keyframing happens, so this can be used in parallel with one of the default camera samplers.


    Example:

    .. code-block:: yaml

        {
          "module": "camera.CameraOffset",
          "config": {
            "target": "Tracker"
          }
        }

    Result: Each frame/rendered image the camera gets moved so that the object named Tracker is at the center.

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "target", "Name of the object to center to."
    """
    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):

        def handler_offset_cam(scene, depsgraph):
            """
            Moves the camera to be centered on the target position.
            """

            nonlocal self

            pos = self.get_pos(depsgraph)
            cam = self.get_cam(depsgraph)
            cam.matrix_world.translation += pos


        bpy.types.RenderSettings.use_lock_interface = True
        bpy.app.handlers.frame_change_post.append(handler_offset_cam)

    def get_cam(self, depsgraph):
        """
        Returns: The camera, evaluated for the current frame.
        """

        cam = bpy.context.scene.camera
        return cam.evaluated_get(depsgraph)

    def get_pos(self, depsgraph):
        """
        Returns: The position of the target object, evaluated for the current frame.
        """

        o = bpy.context.scene.objects[self.config.data["target"]]
        eo = o.evaluated_get(depsgraph)

        return eo.matrix_world.translation

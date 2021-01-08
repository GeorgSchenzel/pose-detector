from pathlib import Path

import bpy

from src.main.Module import Module


class RenderFilePathOverwriter(Module):
    """
    Sets the prefix of each rendered file to the value of a custom property on an object.


    Example:

    .. code-block:: yaml

        {
          "module": "renderer.RenderFilePathOverwriter",
          "config": {
            "entity": "Armature",
            "property": "open"
          }
        }

    Result: Each file gets a prefix containing the value of the property "open" the entity called "Armature" had at
        the time of rendering.

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "entity", "Name of the entity to read from."
        "property", "Name of the custom property to read."
    """
    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):

        def handler_render_init(scene):
            """
            This needs to run after the modules are initialized. To get the target render path.
            """

            nonlocal self

            self.base_path = Path(bpy.context.scene.render.filepath).parent

        def handler_set_filepath(scene, depsgraph):
            """
            Sets the target value as a prefix to the rendered file.
            """
            nonlocal self

            print(scene.frame_current, bpy.context.scene.frame_current)
            prefix = "{}_".format(self.get_value(depsgraph))
            scene.render.filepath = str(self.base_path.joinpath(prefix))


        bpy.types.RenderSettings.use_lock_interface = True
        bpy.app.handlers.render_init.append(handler_render_init)
        bpy.app.handlers.frame_change_post.append(handler_set_filepath)

    def get_value(self, depsgraph):
        """
        Returns: The value of the custom property in this frame.
        """

        o = bpy.context.scene.objects[self.config.data["entity"]]
        eo = o.evaluated_get(depsgraph)
        value = eo[self.config.data["property"]]

        print(value)
        return str(round(value * 100))
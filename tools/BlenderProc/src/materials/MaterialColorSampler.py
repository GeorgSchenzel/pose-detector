import warnings

from src.main.Module import Module
from src.utility.Config import Config
from src.utility.Utility import Utility


class MaterialColorSampler(Module):
    """ Sample colors and assign them as base_color to materials every frame.


    Example:

    .. code-block:: yaml

        {
          "module": "materials.MaterialColorSampler",
          "config": {
            "selector": {
              "provider": "getter.Material",
              "conditions": {
                "name": "Material"
              }
            },
            "color": [1, 0, 0, 1]
          }
        }

    Result: The material called "Material" will be set to a red color.

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "number_of_samples", "The number of times a color is sampled. Type: int. Default: 1."
        "selector", "Materials to become subjects of manipulation. Type: Provider."
        "color", "The color to be used, provided as a RGBA vector. Type: Array"
    """

    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):
        """ Samples colors and assigns them to selected materials over multiple frames.
            1. Select materials.
            2. For each frame set all materials to a new sampled color.
        """
        set_params = {}
        sel_objs = {}
        for key in self.config.data.keys():
            # if its not a selector -> to the set parameters dict
            if key != 'selector':
                set_params[key] = self.config.data[key]
            else:
                sel_objs[key] = self.config.data[key]
        # create Config objects
        params_conf = Config(set_params)
        sel_conf = Config(sel_objs)
        # invoke a Getter, get a list of entities to manipulate
        materials = sel_conf.get_list("selector")

        number_of_samples = self.config.get_int("number_of_samples", 1)

        if not materials:
            warnings.warn("Warning: No materials selected inside of the MaterialManipulator")
            return

        for frame_id in range(1, number_of_samples + 1):
            for material in materials:
                if not material.use_nodes:
                    raise Exception("This material does not use nodes -> not supported here.")

                color = params_conf.get_raw_value("color")
                self._set_principled_shader_value(material, "base_color", color, frame_id)


    @staticmethod
    def _set_principled_shader_value(material, shader_input_key, value, frame_id):
        """

        :param material: Material to be modified. Type: bpy.types.Material.
        :param shader_input_key: Name of the shader's input. Type: string.
        :param value: Value to set.
        """
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled_bsdf = Utility.get_the_one_node_with_type(nodes, "BsdfPrincipled")
        shader_input_key_copy = shader_input_key.replace("_", " ").title()
        if principled_bsdf.inputs[shader_input_key_copy].links:
            links.remove(principled_bsdf.inputs[shader_input_key_copy].links[0])
        if shader_input_key_copy in principled_bsdf.inputs:
            principled_bsdf.inputs[shader_input_key_copy].default_value = value
        else:
            raise Exception("Shader input key '{}' is not a part of the shader.".format(shader_input_key_copy))

        # add keyframes for changed value
        principled_bsdf.inputs[shader_input_key_copy].keyframe_insert(data_path='default_value', frame=frame_id)

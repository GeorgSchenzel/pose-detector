import warnings

import bpy

from src.main.Module import Module
from src.utility.Config import Config


class EntityCustomPropertySampler(Module):
    """ Performs manipulation of custom properties on selected entities of different Blender built-in types, e.g. Mesh objects, Camera
        objects, Light objects, etc. The this is similar to the EntityManipulator, with the difference being that
        the value is sampled each frame instead of once per batch.

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "selector", "Objects to become subjects of manipulation. Type: Provider."

    **Values to set**:

    .. csv-table::
        :header: "Parameter", "Description"

        "key", "Name of the custom property to change."
               "Type: string. "
        "value", "Value of the attribute/custom prop. to set or input value(s) for a custom function. Type: string, "
                 "int, bool or float, list/Vector."

    """
    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):
        """
            Sets according values of defined custom properties to the selected
            entities.
        """
        # separating defined part with the selector from ambiguous part with attribute names and their values to set
        set_params = {}
        sel_objs = {}
        for key in self.config.data.keys():
            if key != 'selector' and key != "mode":
                # if its not a selector -> to the set parameters dict
                set_params[key] = self.config.data[key]
            else:
                sel_objs[key] = self.config.data[key]
        # create Config objects
        params_conf = Config(set_params)
        sel_conf = Config(sel_objs)
        # invoke a Getter, get a list of entities to manipulate
        entities = sel_conf.get_list("selector")

        number_of_samples = self.config.get_int("number_of_samples", 1)

        if not entities:
            warnings.warn("Warning: No entities are selected. Check Providers conditions.")
            return
        else:
            print("Amount of objects to modify: {}.".format(len(entities)))

        for frame_id in range(number_of_samples):
            params = self._get_the_set_params(params_conf)
            for key, value in params.items():
                for entity in entities:

                    # used so we don't modify original key when having more than one entity
                    key_copy = key

                    if hasattr(entity, key_copy):
                        setattr(entity, key_copy, value)
                        entity.keyframe_insert(data_path=key_copy, frame=frame_id)

                    else:
                        entity[key_copy] = value
                        entity.keyframe_insert(data_path='["{}"]'.format(key_copy), frame=frame_id)

        bpy.context.view_layer.update()

    def _get_the_set_params(self, params_conf):
        """ Extracts actual values to set from a Config object.

        :param params_conf: Object with all user-defined data. Type: Config.
        :return: Parameters to set as {name of the parameter: it's value} pairs. Type: dict.
        """
        params = {}
        for key in params_conf.data.keys():
            result = params_conf.get_raw_value(key)

            params.update({key: result})

        return params

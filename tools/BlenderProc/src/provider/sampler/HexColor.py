import random

import mathutils

from src.main.Provider import Provider


class HexColor(Provider):
    """ Samples a 4-dimensional RGBA vector from a list of hex color strings.


    Example 1:

        {
          "provider": "sampler.Color",
          "options": [
                "#ff0000",
                "#00ff00",
                "#0000ff"
            ]
        }

    Result: One of the colors is selected and converted to a vector.

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "options", "A list of hex color strings to choose from."
    """

    def __init__(self, config):
        Provider.__init__(self, config)

    def run(self):
        """ Samples a RGBA vector from the provided options.

        :return: RGBA vector. Type: mathutils.Vector
        """
        # options
        colors = self.config.get_list("options")

        color = random.choice(colors).lstrip('#')
        color = [srgb_to_linearrgb(parse_color(color[i:i + 2])) for i in (0, 2, 4)] + [1]
        color = mathutils.Vector(color)

        return color


def parse_color(hex):
    """
    Reads a hex two digit hex string and converts it to a decimal between 0 and 1.
    """
    return int(hex, 16) / 255.


def srgb_to_linearrgb(c):
    """
    This color conversion needs to be done to match the target color space blender uses.
    """
    if c < 0:
        return 0
    elif c < 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4
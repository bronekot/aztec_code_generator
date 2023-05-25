#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    aztec_code_generator
    ~~~~~~~~~~~~~~~~~~~~

    Aztec code generator.

    :copyright: (c) 2016-2022 by Dmitry Alimov.
    :license: The MIT License (MIT), see LICENSE for more details.
"""

import math
import numbers
import sys
from azteccode_class import AztecCode

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = ImageDraw = None
    missing_pil = sys.exc_info()

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from tables import *
from aztecfunctions import *






def main():
    data = "Aztec Code 2D :) ec=50"
    aztec_code = AztecCode(data, size=None, compact=None, ec_percent=50)
    aztec_code.print_out()
    if ImageDraw is None:
        print("PIL is not installed, cannot generate PNG")
    else:
        aztec_code.save("aztec_code.png", 4)
    print(
        "Aztec Code info: {0}x{0} {1} {2}".format(
            aztec_code.size,
            "(compact)" if aztec_code.compact else "",
            aztec_code.ec_percent,
        )
    )


if __name__ == "__main__":
    main()

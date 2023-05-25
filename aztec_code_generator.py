#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    aztec_code_generator
    ~~~~~~~~~~~~~~~~~~~~

    Aztec code generator.

    :copyright: (c) 2016-2022 by Dmitry Alimov.
    :license: The MIT License (MIT), see LICENSE for more details.

    Forked and improved by Andrei Dziuba.
    - Added support for selecting minimum error correction level.
    - Increased performance.
"""


from azteccode_class import AztecCode
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

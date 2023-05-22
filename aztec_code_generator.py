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


def prod(x, y, log, alog, gf):
    """Product x times y."""
    return 0 if not x or not y else alog[(log[x] + log[y]) % (gf - 1)]


def reed_solomon(wd, nd, nc, gf, pp):
    """Calculate error correction codewords.

    Algorithm is based on Aztec Code bar code symbology specification from
    GOST-R-ISO-MEK-24778-2010 (Russian)
    Takes ``nd`` data codeword values in ``wd`` and adds on ``nc`` check
    codewords, all within GF(gf) where ``gf`` is a power of 2 and ``pp``
    is the value of its prime modulus polynomial.

    :param list[int] wd: Data codewords (in/out param).
    :param int nd: Number of data codewords.
    :param int nc: Number of error correction codewords.
    :param int gf: Galois Field order.
    :param int pp: Prime modulus polynomial value.

    :return: None.
    """
    # generate log and anti log tables
    log = {0: 1 - gf}
    alog = {0: 1}
    for i in range(1, gf):
        alog[i] = alog[i - 1] * 2
        if alog[i] >= gf:
            alog[i] ^= pp
        log[alog[i]] = i
    # generate polynomial coefficients
    c = {0: 1}
    for i in range(1, nc + 1):
        c[i] = 0
    for i in range(1, nc + 1):
        c[i] = c[i - 1]
        for j in range(i - 1, 0, -1):
            c[j] = c[j - 1] ^ prod(c[j], alog[i], log, alog, gf)
        c[0] = prod(c[0], alog[i], log, alog, gf)
    # generate codewords
    for i in range(nd, nd + nc):
        wd[i] = 0
    for i in range(nd):
        assert 0 <= wd[i] < gf
        k = wd[nd] ^ wd[i]
        for j in range(nc):
            wd[nd + j] = prod(k, c[nc - j - 1], log, alog, gf)
            if j < nc - 1:
                wd[nd + j] ^= wd[nd + j + 1]


def find_optimal_sequence(data):
    """Find optimal sequence, i.e. with minimum number of bits to encode data.

    TODO: add support of FLG(n) processing

    :param list[str|int] data: Data to encode.

    :return: Optimal sequence.
    """
    back_to = {
        "upper": "upper",
        "lower": "upper",
        "mixed": "upper",
        "punct": "upper",
        "digit": "upper",
        "binary": "upper",
    }
    cur_len = {"upper": 0, "lower": E, "mixed": E, "punct": E, "digit": E, "binary": E}
    cur_seq = {
        "upper": [],
        "lower": [],
        "mixed": [],
        "punct": [],
        "digit": [],
        "binary": [],
    }
    prev_c = ""
    for c in data:
        for x in modes:
            for y in modes:
                if cur_len[x] + latch_len[x][y] < cur_len[y]:
                    cur_len[y] = cur_len[x] + latch_len[x][y]
                    if y == "binary":
                        # for binary mode use B/S instead of B/L
                        if x in ["punct", "digit"]:
                            # if changing from punct or digit to binary mode use U/L as intermediate mode
                            # TODO: update for digit
                            back_to[y] = "upper"
                            cur_seq[y] = cur_seq[x] + [
                                "U/L",
                                "%s/S" % y.upper()[0],
                                "size",
                            ]
                        else:
                            back_to[y] = x
                            cur_seq[y] = cur_seq[x] + ["%s/S" % y.upper()[0], "size"]
                    else:
                        if cur_seq[x]:
                            # if changing from punct or digit mode - use U/L as intermediate mode
                            # TODO: update for digit
                            if x in ["punct", "digit"] and y != "upper":
                                cur_seq[y] = cur_seq[x] + [
                                    "resume",
                                    "U/L",
                                    "%s/L" % y.upper()[0],
                                ]
                                back_to[y] = y
                            elif x in ["upper", "lower"] and y == "punct":
                                cur_seq[y] = cur_seq[x] + ["M/L", "%s/L" % y.upper()[0]]
                                back_to[y] = y
                            elif x == "mixed" and y != "upper":
                                if y == "punct":
                                    cur_seq[y] = cur_seq[x] + ["P/L"]
                                    back_to[y] = "punct"
                                else:
                                    cur_seq[y] = cur_seq[x] + ["U/L", "D/L"]
                                    back_to[y] = "digit"
                                continue
                            else:
                                if x == "binary":
                                    # TODO: review this
                                    # Reviewed by jravallec
                                    if y == back_to[x]:
                                        # when return from binary to previous mode, skip mode change
                                        cur_seq[y] = cur_seq[x] + ["resume"]
                                    elif y == "upper":
                                        if back_to[x] == "lower":
                                            cur_seq[y] = cur_seq[x] + [
                                                "resume",
                                                "M/L",
                                                "U/L",
                                            ]
                                        if back_to[x] == "mixed":
                                            cur_seq[y] = cur_seq[x] + ["resume", "U/L"]
                                        back_to[y] = "upper"
                                    elif y == "lower":
                                        cur_seq[y] = cur_seq[x] + ["resume", "L/L"]
                                        back_to[y] = "lower"
                                    elif y == "mixed":
                                        cur_seq[y] = cur_seq[x] + ["resume", "M/L"]
                                        back_to[y] = "mixed"
                                    elif y == "punct":
                                        if back_to[x] == "mixed":
                                            cur_seq[y] = cur_seq[x] + ["resume", "P/L"]
                                        else:
                                            cur_seq[y] = cur_seq[x] + [
                                                "resume",
                                                "M/L",
                                                "P/L",
                                            ]
                                        back_to[y] = "punct"
                                    elif y == "digit":
                                        if back_to[x] == "mixed":
                                            cur_seq[y] = cur_seq[x] + [
                                                "resume",
                                                "U/L",
                                                "D/L",
                                            ]
                                        else:
                                            cur_seq[y] = cur_seq[x] + ["resume", "D/L"]
                                        back_to[y] = "digit"
                                else:
                                    cur_seq[y] = cur_seq[x] + [
                                        "resume",
                                        "%s/L" % y.upper()[0],
                                    ]
                                    back_to[y] = y
                        else:
                            # if changing from punct or digit mode - use U/L as intermediate mode
                            # TODO: update for digit
                            if x in ["punct", "digit"]:
                                cur_seq[y] = cur_seq[x] + ["U/L", "%s/L" % y.upper()[0]]
                                back_to[y] = y
                            elif x in ["binary", "upper", "lower"] and y == "punct":
                                cur_seq[y] = cur_seq[x] + ["M/L", "%s/L" % y.upper()[0]]
                                back_to[y] = y
                            else:
                                cur_seq[y] = cur_seq[x] + ["%s/L" % y.upper()[0]]
                                back_to[y] = y
        next_len = {
            "upper": E,
            "lower": E,
            "mixed": E,
            "punct": E,
            "digit": E,
            "binary": E,
        }
        next_seq = {
            "upper": [],
            "lower": [],
            "mixed": [],
            "punct": [],
            "digit": [],
            "binary": [],
        }
        possible_modes = []
        if c in upper_chars:
            possible_modes.append("upper")
        if c in lower_chars:
            possible_modes.append("lower")
        if c in mixed_chars:
            possible_modes.append("mixed")
        if c in punct_chars:
            possible_modes.append("punct")
        if c in digit_chars:
            possible_modes.append("digit")
        possible_modes.append("binary")
        for x in possible_modes:
            # TODO: review this!
            if back_to[x] == "digit" and x == "lower":
                cur_seq[x] = cur_seq[x] + ["U/L", "L/L"]
                cur_len[x] = cur_len[x] + latch_len[back_to[x]][x]
                back_to[x] = "lower"
            # add char to current sequence
            if cur_len[x] + char_size[x] < next_len[x]:
                next_len[x] = cur_len[x] + char_size[x]
                next_seq[x] = cur_seq[x] + [c]
            for y in modes[:-1]:
                if y == x:
                    continue
                if cur_len[y] + shift_len[y][x] + char_size[x] < next_len[y]:
                    next_len[y] = cur_len[y] + shift_len[y][x] + char_size[x]
                    next_seq[y] = cur_seq[y] + ["%s/S" % x.upper()[0]] + [c]
        # TODO: review this!!!
        if prev_c and prev_c + c in punct_2_chars:
            for x in modes:
                last_mode = ""
                for char in cur_seq[x][::-1]:
                    if char.replace("/S", "").replace("/L", "") in abbr_modes:
                        last_mode = abbr_modes.get(
                            char.replace("/S", "").replace("/L", "")
                        )
                        break
                if (
                    last_mode == "punct"
                    and (cur_seq[x][-1] + c in punct_2_chars and x != "mixed")
                    and cur_len[x] < next_len[x]
                ):
                    next_len[x] = cur_len[x]
                    next_seq[x] = cur_seq[x][:-1] + [cur_seq[x][-1] + c]
        if len(next_seq["binary"]) - 2 == 32:
            next_len["binary"] += 11
        for i in modes:
            cur_len[i] = next_len[i]
            cur_seq[i] = next_seq[i]
        prev_c = c
    # sort in ascending order and get shortest sequence
    result_seq = []
    sorted_cur_len = sorted(cur_len, key=cur_len.get)
    if sorted_cur_len:
        min_length = sorted_cur_len[0]
        result_seq = cur_seq[min_length]
    # update binary sequences' sizes
    sizes = {}
    result_seq_len = len(result_seq)
    reset_pos = result_seq_len - 1
    for i, c in enumerate(result_seq[::-1]):
        if c == "size":
            sizes[i] = reset_pos - (result_seq_len - i - 1)
            reset_pos = result_seq_len - i
        elif c == "resume":
            reset_pos = result_seq_len - i - 2
    for size_pos in sizes:
        result_seq[len(result_seq) - size_pos - 1] = sizes[size_pos]
    # remove 'resume' tokens
    result_seq = [x for x in result_seq if x != "resume"]
    # update binary sequences' extra sizes
    updated_result_seq = []
    is_binary_length = False
    for i, c in enumerate(result_seq):
        if is_binary_length:
            if c > 31:
                updated_result_seq.append(0)
                updated_result_seq.append(c - 31)
            else:
                updated_result_seq.append(c)
            is_binary_length = False
        else:
            updated_result_seq.append(c)

        if c == "B/S":
            is_binary_length = True

    return updated_result_seq


def optimal_sequence_to_bits(optimal_sequence):
    """Convert optimal sequence to bits.

    :param list[str|int] optimal_sequence: Input optimal sequence.

    :return: String with bits.
    """
    out_bits = ""
    mode = "upper"
    prev_mode = "upper"
    shift = False
    binary = False
    binary_seq_len = 0
    binary_index = 0
    sequence = optimal_sequence[:]
    while not not sequence:
        ch = sequence.pop(0)
        if binary:
            out_bits += bin(ord(ch))[2:].zfill(char_size.get(mode))
            binary_index += 1
            # resume previous mode at the end of the binary sequence
            if binary_index >= binary_seq_len:
                mode = prev_mode
                binary = False
            continue
        index = code_chars.get(mode).index(ch)
        out_bits += bin(index)[2:].zfill(char_size.get(mode))
        # resume previous mode for shift
        if shift:
            mode = prev_mode
            shift = False
        # get mode from sequence character
        if ch.endswith("/L"):
            mode = abbr_modes.get(ch.replace("/L", ""))
        elif ch.endswith("/S"):
            mode = abbr_modes.get(ch.replace("/S", ""))
            shift = True
        # handle binary mode
        if mode == "binary":
            if not sequence:
                raise Exception("Expected binary sequence length")
            # followed by a 5 bit length
            seq_len = sequence.pop(0)
            if not isinstance(seq_len, numbers.Number):
                raise Exception("Binary sequence length must be a number")
            out_bits += bin(seq_len)[2:].zfill(5)
            binary_seq_len = seq_len
            # if length is zero - 11 additional length bits are used for length
            if not binary_seq_len:
                seq_len = sequence.pop(0)
                if not isinstance(seq_len, numbers.Number):
                    raise Exception("Binary sequence length must be a number")
                out_bits += bin(seq_len)[2:].zfill(11)
                binary_seq_len = seq_len + 31
            binary = True
            binary_index = 0
        # update previous mode
        if not shift:
            prev_mode = mode
    return out_bits


def get_data_codewords(bits, codeword_size):
    """Get codewords stream from data bits sequence.
    Bit stuffing and padding are used to avoid all-zero and all-ones codewords.

    :param str bits: Input data bits.
    :param int codeword_size: Codeword size in bits.

    :return: Data codewords.
    """
    codewords = []
    sub_bits = ""
    for bit in bits:
        sub_bits += bit
        # if first bits of sub sequence are zeros add 1 as a last bit
        if len(sub_bits) == codeword_size - 1 and sub_bits.find("1") < 0:
            sub_bits += "1"
        # if first bits of sub sequence are ones add 0 as a last bit
        if len(sub_bits) == codeword_size - 1 and sub_bits.find("0") < 0:
            sub_bits += "0"
        # convert bits to decimal int and add to result codewords
        if len(sub_bits) >= codeword_size:
            codewords.append(int(sub_bits, 2))
            sub_bits = ""
    if sub_bits:
        # update and add final bits
        sub_bits = sub_bits.ljust(codeword_size, "1")
        # change final bit to zero if all bits are ones
        if sub_bits.find("0") < 0:
            sub_bits = sub_bits[:-1] + "0"
        codewords.append(int(sub_bits, 2))
    return codewords


def get_config_from_table(size, compact):
    """Get config from table with given size and compactness flag.

    :param int size: Matrix size.
    :param bool compact: Compactness flag.

    :return: Dict with config.
    """
    config = table.get((size, compact))
    if not config:
        raise Exception("Failed to find config with size and compactness flag")
    return config


class AztecCode(object):
    def __init__(self, data, size=None, compact=None, ec_percent=23):
        """
        Initializes an instance of the QRCode class with the given data, size, compact, and error correction percentage.

        Args:
            data: The data to be encoded into the QR code.
            size: The size of the QR code matrix. If not provided, the optimal size is chosen automatically.
            compact: A boolean indicating whether to use the compact mode or not. If not provided, the optimal mode is chosen automatically.
            ec_percent: The error correction percentage. Must be between 5 and 95. Defaults to 23 if not provided.

        Raises:
            Exception: If the provided size and compact values are not found in the sizes table.

        Returns:
            None
        """
        self.data = data
        self.ec_percent = max(5, min(ec_percent, 95))
        # If size and compact parameters are given, check if they're valid.
        if size is not None and compact is not None:
            if (size, compact) in table:
                self.size, self.compact = size, compact
            else:
                raise Exception(
                    "Given size and compact values (%s, %s) are not found in sizes table!"
                    % (size, compact)
                )
        # If size and compact parameters are not given, find the optimal values.
        else:
            self.size, self.compact = self.find_suitable_matrix_size()
        self.__create_matrix()
        self.__encode_data()

    def __create_matrix(self):
        """Create Aztec code matrix with given size."""
        self.matrix = []
        for _ in range(self.size):
            line = []
            for __ in range(self.size):
                line.append(" ")
            self.matrix.append(line)

    def find_suitable_matrix_size(self):
        """Find suitable matrix size.
        Raise an exception if suitable size is not found.

        :param list[str|int] data: Data to encode.

        :return: (size, compact) tuple.
        """
        optimal_sequence = find_optimal_sequence(self.data)
        out_bits = optimal_sequence_to_bits(optimal_sequence)
        for size, compact in sorted(table.keys()):
            config = get_config_from_table(size, compact)
            bits = config.get("bits")
            # calculate minimum required number of bits
            required_bits_count = int(
                math.ceil(
                    len(out_bits) * 100.0 / (100 - self.ec_percent)
                    + 3 * 100.0 / (100 - self.ec_percent)
                )
            )
            if required_bits_count < bits:
                return size, compact
        raise Exception("Data too big to fit in one Aztec code!")

    def save(self, filename, module_size=1):
        """Save matrix to image file.

        :param str filename: Output image filename.
        :param int module_size: Barcode module size in pixels.

        :return: None.
        """
        if ImageDraw is None:
            exc = missing_pil[0](missing_pil[1])
            exc.__traceback__ = missing_pil[2]
            raise exc
        image = Image.new(
            "RGB", (self.size * module_size, self.size * module_size), "white"
        )
        image_draw = ImageDraw.Draw(image)
        for y in range(self.size):
            for x in range(self.size):
                image_draw.rectangle(
                    (
                        x * module_size,
                        y * module_size,
                        x * module_size + module_size,
                        y * module_size + module_size,
                    ),
                    fill=(0, 0, 0) if self.matrix[y][x] == "#" else (255, 255, 255),
                )
        image.save(filename)

    def print_out(self):
        """Print out Aztec code matrix."""
        for line in self.matrix:
            print("".join(x for x in line))

    def __add_finder_pattern(self):
        """Add bulls-eye finder pattern."""
        center = self.size // 2
        ring_radius = 5 if self.compact else 7
        for x in range(-ring_radius, ring_radius):
            for y in range(-ring_radius, ring_radius):
                if (max(abs(x), abs(y)) + 1) % 2:
                    self.matrix[center + y][center + x] = "#"

    def __add_orientation_marks(self):
        """Add orientation marks to matrix."""
        center = self.size // 2
        ring_radius = 5 if self.compact else 7
        # add orientation marks
        # left-top
        self.matrix[center - ring_radius][center - ring_radius] = "#"
        self.matrix[center - ring_radius + 1][center - ring_radius] = "#"
        self.matrix[center - ring_radius][center - ring_radius + 1] = "#"
        # right-top
        self.matrix[center - ring_radius + 0][center + ring_radius + 0] = "#"
        self.matrix[center - ring_radius + 1][center + ring_radius + 0] = "#"
        # right-down
        self.matrix[center + ring_radius - 1][center + ring_radius + 0] = "#"

    def __add_reference_grid(self):
        """Add reference grid to matrix."""
        if self.compact:
            return
        center = self.size // 2
        ring_radius = 5 if self.compact else 7
        for x in range(-center, center + 1):
            for y in range(-center, center + 1):
                # skip finder pattern
                if (
                    -ring_radius <= x <= ring_radius
                    and -ring_radius <= y <= ring_radius
                ):
                    continue
                # set pixel
                if x % 16 == 0 or y % 16 == 0:
                    val = "#" if (x + y + 1) % 2 != 0 else " "
                    self.matrix[center + y][center + x] = val

    def __get_mode_message(self, layers_count, data_cw_count):
        """Get mode message.

        :param int layers_count: Number of layers.
        :param int data_cw_count: Number of data codewords.

        :return: Mode message codewords.
        """
        if self.compact:
            # for compact mode - 2 bits with layers count and 6 bits with data codewords count
            mode_word = "{0:02b}{1:06b}".format(layers_count - 1, data_cw_count - 1)
            # two 4 bits initial codewords with 5 Reed-Solomon check codewords
            init_codewords = [int(mode_word[i : i + 4], 2) for i in range(0, 8, 4)]
            total_cw_count = 7
        else:
            # for full mode - 5 bits with layers count and 11 bits with data codewords count
            mode_word = "{0:05b}{1:011b}".format(layers_count - 1, data_cw_count - 1)
            # four 4 bits initial codewords with 6 Reed-Solomon check codewords
            init_codewords = [int(mode_word[i : i + 4], 2) for i in range(0, 16, 4)]
            total_cw_count = 10
        # fill Reed-Solomon check codewords with zeros
        init_cw_count = len(init_codewords)
        codewords = (init_codewords + [0] * (total_cw_count - init_cw_count))[
            :total_cw_count
        ]
        # update Reed-Solomon check codewords using GF(16)
        reed_solomon(
            codewords, init_cw_count, total_cw_count - init_cw_count, 16, polynomials[4]
        )
        return codewords

    def __add_mode_info(self, data_cw_count):
        """Add mode info to matrix.

        :param int data_cw_count: Number of data codewords.

        :return: None.
        """
        config = get_config_from_table(self.size, self.compact)
        layers_count = config.get("layers")
        mode_data_values = self.__get_mode_message(layers_count, data_cw_count)
        mode_data_bits = "".join("{0:04b}".format(v) for v in mode_data_values)

        center = self.size // 2
        ring_radius = 5 if self.compact else 7
        side_size = 7 if self.compact else 11
        bits_stream = StringIO(mode_data_bits)
        x = 0
        y = 0
        index = 0
        while True:
            # for full mode take a reference grid into account
            if not self.compact and (index % side_size) == 5:
                index += 1
                continue
            # read one bit
            bit = bits_stream.read(1)
            if not bit:
                break
            if 0 <= index < side_size:
                # top
                x = index + 2 - ring_radius
                y = -ring_radius
            elif side_size <= index < side_size * 2:
                # right
                x = ring_radius
                y = index % side_size + 2 - ring_radius
            elif side_size * 2 <= index < side_size * 3:
                # bottom
                x = ring_radius - index % side_size - 2
                y = ring_radius
            elif side_size * 3 <= index < side_size * 4:
                # left
                x = -ring_radius
                y = ring_radius - index % side_size - 2
            # set pixel
            self.matrix[center + y][center + x] = "#" if bit == "1" else " "
            index += 1

    def __add_data(self):
        """Add data to encode to the matrix.

        :param list[str|int] data: data to encode.

        :return: number of data codewords.
        """
        optimal_sequence = find_optimal_sequence(self.data)
        out_bits = optimal_sequence_to_bits(optimal_sequence)
        config = get_config_from_table(self.size, self.compact)
        layers_count = config.get("layers")
        cw_count = config.get("codewords")
        cw_bits = config.get("cw_bits")
        bits = config.get("bits")

        # calculate minimum required number of bits
        required_bits_count = int(
            math.ceil(
                len(out_bits) * 100.0 / (100 - self.ec_percent)
                + 3 * 100.0 / (100 - self.ec_percent)
            )
        )
        data_codewords = get_data_codewords(out_bits, cw_bits)
        if required_bits_count > bits:
            raise Exception("Data too big to fit in Aztec code with current size!")

        # add Reed-Solomon codewords to init data codewords
        data_cw_count = len(data_codewords)
        codewords = (data_codewords + [0] * (cw_count - data_cw_count))[:cw_count]
        reed_solomon(
            codewords,
            data_cw_count,
            cw_count - data_cw_count,
            2**cw_bits,
            polynomials[cw_bits],
        )

        center = self.size // 2
        ring_radius = 5 if self.compact else 7

        num = 2
        side = "top"
        layer_index = 0
        pos_x = center - ring_radius
        pos_y = center - ring_radius - 1
        full_bits = "".join(bin(cw)[2:].zfill(cw_bits) for cw in codewords)[::-1]
        for i in range(0, len(full_bits), 2):
            num += 1
            max_num = ring_radius * 2 + layer_index * 4 + (4 if self.compact else 3)
            bits_pair = ["#" if bit == "1" else " " for bit in full_bits[i : i + 2]]
            if layer_index >= layers_count:
                raise Exception("Maximum layer count for current size is exceeded!")
            if side == "top":
                # move right
                dy0 = 1 if not self.compact and (center - pos_y) % 16 == 0 else 0
                dy1 = 2 if not self.compact and (center - pos_y + 1) % 16 == 0 else 1
                self.matrix[pos_y - dy0][pos_x] = bits_pair[0]
                self.matrix[pos_y - dy1][pos_x] = bits_pair[1]
                pos_x += 1
                if num > max_num:
                    num = 2
                    side = "right"
                    pos_x -= 1
                    pos_y += 1
                # skip reference grid
                if not self.compact and (center - pos_x) % 16 == 0:
                    pos_x += 1
                if not self.compact and (center - pos_y) % 16 == 0:
                    pos_y += 1
            elif side == "right":
                # move down
                dx0 = 1 if not self.compact and (center - pos_x) % 16 == 0 else 0
                dx1 = 2 if not self.compact and (center - pos_x + 1) % 16 == 0 else 1
                self.matrix[pos_y][pos_x - dx0] = bits_pair[1]
                self.matrix[pos_y][pos_x - dx1] = bits_pair[0]
                pos_y += 1
                if num > max_num:
                    num = 2
                    side = "bottom"
                    pos_x -= 2
                    if not self.compact and (center - pos_x - 1) % 16 == 0:
                        pos_x -= 1
                    pos_y -= 1
                # skip reference grid
                if not self.compact and (center - pos_y) % 16 == 0:
                    pos_y += 1
                if not self.compact and (center - pos_x) % 16 == 0:
                    pos_x -= 1
            elif side == "bottom":
                # move left
                dy0 = 1 if not self.compact and (center - pos_y) % 16 == 0 else 0
                dy1 = 2 if not self.compact and (center - pos_y + 1) % 16 == 0 else 1
                self.matrix[pos_y - dy0][pos_x] = bits_pair[1]
                self.matrix[pos_y - dy1][pos_x] = bits_pair[0]
                pos_x -= 1
                if num > max_num:
                    num = 2
                    side = "left"
                    pos_x += 1
                    pos_y -= 2
                    if not self.compact and (center - pos_y - 1) % 16 == 0:
                        pos_y -= 1
                # skip reference grid
                if not self.compact and (center - pos_x) % 16 == 0:
                    pos_x -= 1
                if not self.compact and (center - pos_y) % 16 == 0:
                    pos_y -= 1
            elif side == "left":
                # move up
                dx0 = 1 if not self.compact and (center - pos_x) % 16 == 0 else 0
                dx1 = 2 if not self.compact and (center - pos_x - 1) % 16 == 0 else 1
                self.matrix[pos_y][pos_x + dx1] = bits_pair[0]
                self.matrix[pos_y][pos_x + dx0] = bits_pair[1]
                pos_y -= 1
                if num > max_num:
                    num = 2
                    side = "top"
                    layer_index += 1
                # skip reference grid
                if not self.compact and (center - pos_y) % 16 == 0:
                    pos_y -= 1
        return data_cw_count

    def __encode_data(self):
        """Encode data."""
        self.__add_finder_pattern()
        self.__add_orientation_marks()
        self.__add_reference_grid()
        data_cw_count = self.__add_data()
        self.__add_mode_info(data_cw_count)


def main():
    data = "Aztec Code 2D :)"
    aztec_code = AztecCode(data)
    aztec_code.print_out()
    if ImageDraw is None:
        print("PIL is not installed, cannot generate PNG")
    else:
        aztec_code.save("aztec_code.png", 4)
    print(
        "Aztec Code info: {0}x{0} {1}".format(
            aztec_code.size, "(compact)" if aztec_code.compact else ""
        )
    )


if __name__ == "__main__":
    main()

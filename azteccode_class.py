import itertools
import math
import sys
from aztecfunctions import *

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


class AztecCode(object):
    def __init__(
        self,
        data,
        size: int | None = None,
        compact: bool | None = None,
        ec_percent: int | None = 23,
    ):
        """
        Initializes an instance of the QRCode class with the given data, size, compact, and error correction percentage.

        Args:
            data: The data to be encoded into the QR code.
            size: The size of the QR code matrix. If not provided, the optimal size is chosen automatically.
            compact: A boolean indicating whether to use the compact mode or not. If not provided, the optimal mode is chosen automatically.
            ec_percent: The minimal error correction percentage. Must be between 5 and 95. Defaults to 23 if not provided.

        Raises:
            Exception: If the provided size and compact values are not found in the sizes table.

        Returns:
            None
        """
        self.data = data
        self.ec_percent = 23 if ec_percent is None else max(5, min(ec_percent, 95))
        # If size and compact parameters are given, check if they're valid.
        if size is None or compact is None:
            self.size, self.compact = self.find_suitable_matrix_size(
                min_size=size, max_size=size, compact_code=compact
            )
        elif (size, compact) in table:
            self.size, self.compact = size, compact
        else:
            raise Exception(
                f"Given size and compact values ({size}, {compact}) are not found in sizes table!"
            )
        self.__create_matrix()
        self.__encode_data()

    def __create_matrix(self):
        """Create Aztec code matrix with given size."""
        self.matrix = []
        for _ in range(self.size):
            line = [" " for __ in range(self.size)]
            self.matrix.append(line)

    def find_suitable_matrix_size(
        self, min_size=None, max_size=None, compact_code=None
    ):
        """Find suitable matrix size.
        Raise an exception if suitable size is not found.

        :param list[str|int] data: Data to encode.

        :return: (size, compact) tuple.
        """
        optimal_sequence = find_optimal_sequence(self.data)
        out_bits = optimal_sequence_to_bits(optimal_sequence)
        # calculate minimum required number of bits
        required_bits_count = int(
            math.ceil(
                len(out_bits) * 100.0 / (100 - self.ec_percent)
                + 3 * 100.0 / (100 - self.ec_percent)
            )
        )
        for size, compact in sorted(table.keys(), key=lambda x: (x[0], -x[1])):
            if compact_code is not None and compact_code != compact:
                continue
            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                raise Exception("Data too big to fit in Aztec code this size!")
            bits = get_config_from_table(size, compact).get("bits")
            if required_bits_count < bits:
                self.ec_percent = self.solve_ec_percent(bits, len(out_bits))

                return size, compact
        raise Exception("Data too big to fit in one Aztec code!")

    def solve_ec_percent(self, bits, L):
        for ec_guess in range(95, 4, -1):
            required_bits_count_guess = int(
                math.ceil(L * 100.0 / (100 - ec_guess) + 3 * 100.0 / (100 - ec_guess))
            )
            if required_bits_count_guess < bits:
                return ec_guess

    def to_pil(self, module_size=1):
        """
        Converts the QR code matrix to a PIL image with the specified module size.

        Args:
            module_size (int): The size of each module in the QR code. Defaults to 1.

        Returns:
            Image: A PIL Image object representing the QR code.
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
        return image

    def save(self, filename, module_size=1):
        """Save matrix to image file.

        :param str filename: Output image filename.
        :param int module_size: Barcode module size in pixels.

        :return: None.
        """
        image = self.to_pil(module_size)
        image.save(filename)

    def print_out(self):
        """Print out Aztec code matrix."""
        for line in self.matrix:
            print("".join(line))

    def __add_finder_pattern(self):
        """Add bulls-eye finder pattern."""
        center = self.size // 2
        ring_radius = 5 if self.compact else 7
        for x, y in itertools.product(
            range(-ring_radius, ring_radius), range(-ring_radius, ring_radius)
        ):
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
        for x, y in itertools.product(
            range(-center, center + 1), range(-center, center + 1)
        ):
            # skip finder pattern
            if -ring_radius <= x <= ring_radius and -ring_radius <= y <= ring_radius:
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

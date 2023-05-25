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


def prod(x, y, log, alog, gf):
    """Product x times y."""
    return 0 if not x or not y else alog[(log[x] + log[y]) % (gf - 1)]

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
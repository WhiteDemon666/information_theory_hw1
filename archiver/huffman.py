# GPT generated
# Двухпроходное сжатие Хаффмана и восстановление файлов формата ZMH1

from collections import Counter
from dataclasses import dataclass
from heapq import heappop, heappush
from math import log2
from pathlib import Path
import os
import struct
import tempfile
import zlib

from probabilities import InputError

MAGIC = b"ZMH1"
HEADER = struct.Struct(">4sQQIH")
CHUNK_SIZE = 65536
MAX_UINT64 = (1 << 64) - 1


class ArchiveError(InputError):
    pass


class ForeignFormatError(ArchiveError):
    pass


class TruncatedArchiveError(ArchiveError):
    pass


class DamagedArchiveError(ArchiveError):
    pass


@dataclass
class ArchiveStats:
    original_bytes: int
    compressed_bytes: int
    header_bytes: int
    data_bytes: int
    entropy: float
    average_length: float

    @property
    def compression_ratio(self) -> float | None:
        # Отношение размера архива к размеру исходного файла
        if self.original_bytes == 0:
            return None
        return self.compressed_bytes / self.original_bytes


def _frequencies(filename: Path) -> tuple[list[int], int, int]:
    # Первый проход: считаем частоты, размер файла и CRC32 исходных данных
    frequencies = [0] * 256
    size = 0
    checksum = 0
    with filename.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            size += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
            for symbol, count in Counter(chunk).items():
                frequencies[symbol] += count
    return frequencies, size, checksum & 0xFFFFFFFF


def _code_lengths(frequencies: list[int]) -> dict[int, int]:
    # Узел дерева: либо байт (int), либо пара дочерних узлов (tuple)
    # Второй элемент очереди обеспечивает однозначный порядок при равных частотах
    heap = []
    for symbol, count in enumerate(frequencies):
        if count:
            heappush(heap, (count, symbol, symbol))

    if not heap:
        return {}
    if len(heap) == 1:
        return {heap[0][1]: 1}

    while len(heap) > 1:
        freq_a, min_a, node_a = heappop(heap)
        freq_b, min_b, node_b = heappop(heap)
        heappush(heap, (freq_a + freq_b, min(min_a, min_b), (node_a, node_b)))

    lengths = {}

    def visit(node, depth):
        if isinstance(node, int):
            lengths[node] = depth
            return
        left, right = node
        visit(left, depth + 1)
        visit(right, depth + 1)

    visit(heap[0][2], 0)
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    # Канонический код полностью определяется длинами кодов каждого байта
    codes = {}
    code = 0
    previous_length = 0
    for symbol, length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= length - previous_length
        if code >= (1 << length):
            raise DamagedArchiveError("Повреждённый файл ZMH: недопустимые длины кодов")
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def _validate_lengths(lengths: dict[int, int]) -> None:
    if any(type(symbol) is not int or not 0 <= symbol <= 255 for symbol in lengths):
        raise DamagedArchiveError("Повреждённый файл ZMH: неверный байт модели")
    if any(type(length) is not int or not 1 <= length <= 255 for length in lengths.values()):
        raise DamagedArchiveError("Повреждённый файл ZMH: неверная длина кода")
    if len(lengths) == 1:
        if next(iter(lengths.values())) != 1:
            raise DamagedArchiveError("Повреждённый файл ZMH: неверная односимвольная модель")
        return
    if len(lengths) > 1:
        max_length = max(lengths.values())
        kraft_sum = sum(1 << (max_length - length) for length in lengths.values())
        if kraft_sum != 1 << max_length:
            raise DamagedArchiveError("Повреждённый файл ZMH: некорректное дерево Хаффмана")


def _paths(input_path, output_path, extension: str, force: bool) -> tuple[Path, Path]:
    source = Path(input_path)
    output = Path(output_path) if output_path is not None else Path(str(source) + extension)
    if not source.is_file():
        raise ArchiveError(f"Не найден входной файл: {source}")
    if source.resolve() == output.resolve():
        raise ArchiveError("Входной и выходной пути совпадают")
    if output.exists() and not force:
        raise ArchiveError(f"Выходной файл уже существует: {output}; используй --force")
    if not output.parent.is_dir():
        raise ArchiveError(f"Не существует каталог назначения: {output.parent}")
    return source, output


def _temporary_path(output: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=".zmh_tmp_", dir=output.parent)
    os.close(fd)
    return Path(name)


def compress_file(input_path, output_path=None, *, force=False) -> ArchiveStats:
    source, output = _paths(input_path, output_path, ".zmh", force)
    frequencies, original_size, checksum = _frequencies(source)
    lengths = _code_lengths(frequencies)
    codes = _canonical_codes(lengths)
    total_bits = sum(frequencies[symbol] * length for symbol, length in lengths.items())
    if original_size > MAX_UINT64 or total_bits > MAX_UINT64:
        raise ArchiveError("Исходный файл слишком большой для формата ZMH1")

    temporary = _temporary_path(output)
    try:
        with temporary.open("wb") as archive:
            # Заголовок: сигнатура, длина исходных данных, число бит, CRC32, число байтов модели
            archive.write(HEADER.pack(MAGIC, original_size, total_bits, checksum, len(lengths)))
            for symbol in sorted(lengths):
                archive.write(bytes((symbol, lengths[symbol])))

            # Второй проход: кодируем каждый байт согласно каноническим кодам
            bits = 0
            accumulator = 0
            written_bits = 0
            seen_size = 0
            seen_crc = 0
            with source.open("rb") as original:
                while chunk := original.read(CHUNK_SIZE):
                    seen_size += len(chunk)
                    seen_crc = zlib.crc32(chunk, seen_crc)
                    buffer = bytearray()
                    for symbol in chunk:
                        if symbol not in codes:
                            raise ArchiveError("Исходный файл изменился между проходами")
                        code, length = codes[symbol]
                        accumulator = (accumulator << length) | code
                        bits += length
                        written_bits += length
                        while bits >= 8:
                            bits -= 8
                            buffer.append((accumulator >> bits) & 255)
                            accumulator &= (1 << bits) - 1
                    archive.write(buffer)

            if seen_size != original_size or (seen_crc & 0xFFFFFFFF) != checksum or written_bits != total_bits:
                raise ArchiveError("Исходный файл изменился между проходами")
            if bits:
                # Неполный последний байт дополняется нулевыми младшими битами
                archive.write(bytes((accumulator << (8 - bits),)))

        # Не перезаписываем существующий файл без явного --force
        if output.exists() and not force:
            raise ArchiveError(f"Выходной файл уже существует: {output}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    entropy = sum((count / original_size) * log2(original_size / count)
                  for count in frequencies if count) if original_size else 0.0
    average_length = total_bits / original_size if original_size else 0.0
    data_bytes = (total_bits + 7) // 8
    header_bytes = HEADER.size + 2 * len(lengths)
    return ArchiveStats(original_size, header_bytes + data_bytes,
                        header_bytes, data_bytes, entropy, average_length)


def _read_exact(source, count: int) -> bytes:
    data = source.read(count)
    if len(data) != count:
        raise TruncatedArchiveError("Обрезанный файл ZMH: недостаточно данных")
    return data


def _read_header(archive):
    prefix = archive.read(len(MAGIC))
    if prefix != MAGIC:
        if len(prefix) < len(MAGIC) and MAGIC.startswith(prefix):
            raise TruncatedArchiveError("Обрезанный файл ZMH: неполная сигнатура")
        raise ForeignFormatError("Чужой формат: отсутствует сигнатура ZMH1")
    original_size, bit_count, checksum, symbol_count = struct.unpack(
        ">QQIH", _read_exact(archive, HEADER.size - len(MAGIC))
    )
    if symbol_count > 256:
        raise DamagedArchiveError("Повреждённый файл ZMH: слишком много символов в модели")
    lengths = {}
    for _ in range(symbol_count):
        symbol, length = _read_exact(archive, 2)
        if symbol in lengths:
            raise DamagedArchiveError("Повреждённый файл ZMH: повторяющийся символ")
        lengths[symbol] = length
    _validate_lengths(lengths)
    if original_size == 0:
        if lengths or bit_count or checksum:
            raise DamagedArchiveError("Повреждённый файл ZMH: неверный пустой архив")
    elif not lengths or bit_count < original_size:
        raise DamagedArchiveError("Повреждённый файл ZMH: несовместимые размеры")
    if len(lengths) == 1 and bit_count != original_size:
        raise DamagedArchiveError("Повреждённый файл ZMH: неверная длина односимвольного потока")
    if lengths and bit_count > original_size * max(lengths.values()):
        raise DamagedArchiveError("Повреждённый файл ZMH: неверное число бит")
    return original_size, bit_count, checksum, lengths


def _default_output_path(input_path) -> Path:
    archive = Path(input_path)
    if archive.suffix.lower() != ".zmh":
        raise ForeignFormatError("Чужой формат: требуется расширение .zmh либо явный --output")
    return archive.with_suffix("")


def decompress_file(input_path, output_path=None, *, force=False) -> ArchiveStats:
    output_path = output_path if output_path is not None else _default_output_path(input_path)
    source, output = _paths(input_path, output_path, "", force)
    temporary = _temporary_path(output)
    try:
        with source.open("rb") as archive, temporary.open("wb") as restored:
            original_size, bit_count, checksum, lengths = _read_header(archive)
            codes = _canonical_codes(lengths)
            reverse = {(length, code): symbol for symbol, (code, length) in codes.items()}
            total_data_bytes = (bit_count + 7) // 8
            remaining_bytes = total_data_bytes
            current_code = 0
            current_length = 0
            used_bits = 0
            restored_size = 0
            restored_crc = 0
            buffer = bytearray()

            while remaining_bytes:
                chunk = _read_exact(archive, min(CHUNK_SIZE, remaining_bytes))
                remaining_bytes -= len(chunk)
                for byte in chunk:
                    bits_in_byte = min(8, bit_count - used_bits)
                    if bits_in_byte < 8 and byte & ((1 << (8 - bits_in_byte)) - 1):
                        raise DamagedArchiveError("Повреждённый файл ZMH: ненулевое дополнение битов")
                    for offset in range(7, 7 - bits_in_byte, -1):
                        current_code = (current_code << 1) | ((byte >> offset) & 1)
                        current_length += 1
                        symbol = reverse.get((current_length, current_code))
                        if symbol is not None:
                            buffer.append(symbol)
                            restored_size += 1
                            current_code = 0
                            current_length = 0
                            if restored_size > original_size:
                                raise DamagedArchiveError("Повреждённый файл ZMH: избыток декодированных данных")
                            if len(buffer) >= CHUNK_SIZE:
                                restored_crc = zlib.crc32(buffer, restored_crc)
                                restored.write(buffer)
                                buffer.clear()
                        elif current_length > 255:
                            raise DamagedArchiveError("Повреждённый файл ZMH: недопустимый префикс")
                    used_bits += bits_in_byte

            if buffer:
                restored_crc = zlib.crc32(buffer, restored_crc)
                restored.write(buffer)

            if archive.read(1):
                raise DamagedArchiveError("Повреждённый файл ZMH: лишние байты после потока")
            if current_length or restored_size != original_size:
                raise DamagedArchiveError("Повреждённый файл ZMH: несогласованная длина потока")
            if (restored_crc & 0xFFFFFFFF) != checksum:
                raise DamagedArchiveError("Повреждённый файл ZMH: контрольная сумма не совпала")

        if output.exists() and not force:
            raise ArchiveError(f"Выходной файл уже существует: {output}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    header_bytes = HEADER.size + 2 * len(lengths)
    data_bytes = (bit_count + 7) // 8
    return ArchiveStats(original_size, header_bytes + data_bytes, header_bytes,
                        data_bytes, 0.0, bit_count / original_size if original_size else 0.0)


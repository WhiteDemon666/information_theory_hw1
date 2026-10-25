# GPT generated

# Этот скрипт создаёт воспроизводимые файлы пяти типов и таблицу измерений
# Он предназначен только для экспериментов, а не для самого кодера Хаффмана
import random
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from archiver import compress_file, decompress_file


def png_chunk(name: bytes, data: bytes) -> bytes:
    body = name + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def make_png() -> bytes:
    # PNG со случайными пикселями: само изображение уже сжато алгоритмом DEFLATE
    rng = random.Random(11)
    width = height = 64
    rows = bytearray()
    for _ in range(height):
        rows.append(0)
        rows.extend(rng.randbytes(width * 3))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + png_chunk(b"IEND", b"")
    )


def main():
    folder = ROOT / "tests" / "measure_data"
    folder.mkdir(parents=True, exist_ok=True)
    samples = {
        "russian.txt": ("Математика и теория информации. " * 4000).encode("utf-8"),
        "english.txt": ("Information theory and lossless compression. " * 4000).encode("utf-8"),
        "compressed_image.png": make_png(),
        "repeated.bin": bytes([0xAB]) * 65536,
        "empty.bin": b"",
    }
    table = [
        "# Измерения задачи 1.3 (сгенерированные тестовые данные)",
        "",
        "| Файл | Исходный, байт | H, бит/байт | Средняя длина, бит/байт | Заголовок, байт | Данные, байт | Архив, байт | Архив / исходный |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in samples.items():
        source = folder / name
        archive = folder / (name + ".zmh")
        restored = folder / (name + ".restored")
        source.write_bytes(data)
        stats = compress_file(source, archive, force=True)
        decompress_file(archive, restored, force=True)
        assert restored.read_bytes() == data
        ratio = "—" if stats.compression_ratio is None else f"{stats.compression_ratio:.6f}"
        table.append(
            f"| `{name}` | {stats.original_bytes} | {stats.entropy:.6f} | "
            f"{stats.average_length:.6f} | {stats.header_bytes} | {stats.data_bytes} | "
            f"{stats.compressed_bytes} | {ratio} |"
        )
    outfile = ROOT / "TASK13_RESULTS.md"
    outfile.write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))
    print(f"\nТаблица сохранена в {outfile}")


if __name__ == "__main__":
    main()

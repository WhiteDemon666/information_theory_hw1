# GPT generated

import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from archiver.huffman import (
    ArchiveError,
    DamagedArchiveError,
    ForeignFormatError,
    TruncatedArchiveError,
    compress_file,
    decompress_file,
)

ROOT = Path(__file__).resolve().parent.parent


class HuffmanArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def roundtrip(self, data: bytes):
        original = self.folder / "test.bin"
        archive = self.folder / "test.bin.zmh"
        restored = self.folder / "restored.bin"
        original.write_bytes(data)
        stats = compress_file(original, archive)
        self.assertEqual(stats.original_bytes, len(data))
        self.assertEqual(stats.header_bytes + stats.data_bytes, stats.compressed_bytes)
        self.assertEqual(stats.compressed_bytes, archive.stat().st_size)
        decompress_file(archive, restored)
        self.assertEqual(restored.read_bytes(), data)
        return stats, archive

    def test_empty_file(self):
        stats, _ = self.roundtrip(b"")
        self.assertEqual(stats.header_bytes, 26)
        self.assertEqual(stats.data_bytes, 0)
        self.assertIsNone(stats.compression_ratio)

    def test_single_repeated_byte(self):
        stats, _ = self.roundtrip(b"X" * 10007)
        self.assertEqual(stats.header_bytes, 28)
        self.assertEqual(stats.average_length, 1.0)
        self.assertEqual(stats.entropy, 0.0)

    def test_all_byte_values(self):
        self.roundtrip(bytes(range(256)) * 9)

    def test_random_data(self):
        rng = random.Random(11)
        self.roundtrip(rng.randbytes(101000))

    def test_ru_en_and_small_binary(self):
        for index, data in enumerate((
            "Русский текст. Сжатие методом Хаффмана!\n".encode("utf-8") * 40,
            b"English text with many repeated words.\n" * 50,
            b"\x00\xff\xfe\x00\xff\x88\x00\x00\x00",
            b"\x00",
            b"ABBAAABBAABBABABAABB",
        )):
            with self.subTest(index=index):
                self.roundtrip(data)
                (self.folder / "restored.bin").unlink()
                (self.folder / "test.bin.zmh").unlink()

    def test_deterministic_archive(self):
        original = self.folder / "input.bin"
        original.write_bytes(b"ABRACADABRA" * 100)
        a = self.folder / "a.zmh"
        b = self.folder / "b.zmh"
        compress_file(original, a)
        compress_file(original, b)
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_wrong_magic(self):
        archive = self.folder / "foreign.zmh"
        archive.write_bytes(b"NOT-ZMH-ARCHIVE")
        with self.assertRaises(ForeignFormatError):
            decompress_file(archive, self.folder / "out")

    def test_truncated_header(self):
        archive = self.folder / "short.zmh"
        archive.write_bytes(b"ZMH1" + b"\x00" * 3)
        with self.assertRaises(TruncatedArchiveError):
            decompress_file(archive, self.folder / "out")

    def test_truncated_payload(self):
        _, archive = self.roundtrip(b"ABABABABAB" * 500)
        damaged = self.folder / "short.zmh"
        damaged.write_bytes(archive.read_bytes()[:-1])
        with self.assertRaises(TruncatedArchiveError):
            decompress_file(damaged, self.folder / "out")

    def test_checksum_detects_bit_flip(self):
        _, archive = self.roundtrip(b"ABABABABAB" * 500)
        raw = bytearray(archive.read_bytes())
        raw[-3] ^= 0b10000000
        damaged = self.folder / "corrupt.zmh"
        damaged.write_bytes(raw)
        with self.assertRaises(DamagedArchiveError):
            decompress_file(damaged, self.folder / "out")
        self.assertFalse((self.folder / "out").exists())

    def test_padding_bits_must_be_zero(self):
        _, archive = self.roundtrip(b"ABABA")
        raw = bytearray(archive.read_bytes())
        raw[-1] |= 0b00000001
        damaged = self.folder / "padding.zmh"
        damaged.write_bytes(raw)
        with self.assertRaises(DamagedArchiveError):
            decompress_file(damaged, self.folder / "out")

    def test_trailing_bytes(self):
        _, archive = self.roundtrip(b"ABABABABAB" * 10)
        damaged = self.folder / "extra.zmh"
        damaged.write_bytes(archive.read_bytes() + b"X")
        with self.assertRaises(DamagedArchiveError):
            decompress_file(damaged, self.folder / "out")

    def test_no_overwrite_by_default(self):
        original = self.folder / "original.txt"
        original.write_bytes(b"hello")
        compress_file(original)
        with self.assertRaises(ArchiveError):
            compress_file(original)
        with self.assertRaises(ArchiveError):
            decompress_file(str(original) + ".zmh")
        decompress_file(str(original) + ".zmh", force=True)
        self.assertEqual(original.read_bytes(), b"hello")

    def test_binary_checksum_field(self):
        _, archive = self.roundtrip(b"abcde" * 20)
        raw = bytearray(archive.read_bytes())
        raw[20] ^= 1  # Один из байтов CRC32 в заголовке.
        damaged = self.folder / "crc.zmh"
        damaged.write_bytes(raw)
        with self.assertRaises(DamagedArchiveError):
            decompress_file(damaged, self.folder / "out")

    def test_corrupted_model(self):
        _, archive = self.roundtrip(b"AB" * 20)
        raw = bytearray(archive.read_bytes())
        raw[27] = 255  # Длина кода первого символа.
        damaged = self.folder / "model.zmh"
        damaged.write_bytes(raw)
        with self.assertRaises(DamagedArchiveError):
            decompress_file(damaged, self.folder / "out")

    def test_cli(self):
        original = self.folder / "text.txt"
        original.write_bytes(b"Hello world\n" * 200)
        archive = self.folder / "text.txt.zmh"
        restored = self.folder / "back.txt"
        compressed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "compress", str(original), "--stats"],
            text=True, capture_output=True, cwd=ROOT, encoding="utf-8"
        )
        self.assertEqual(compressed.returncode, 0, compressed.stderr)
        self.assertIn("Энтропия", compressed.stdout)
        self.assertTrue(archive.exists())
        decompressed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "decompress", str(archive), "--output", str(restored)],
            text=True, capture_output=True, cwd=ROOT, encoding="utf-8"
        )
        self.assertEqual(decompressed.returncode, 0, decompressed.stderr)
        self.assertEqual(restored.read_bytes(), original.read_bytes())


if __name__ == "__main__":
    unittest.main()

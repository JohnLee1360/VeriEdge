from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "artifacts" / "_shared" / "thc" / "src"))

from hetero_transport import decode_array, encode_array


class HeteroTransportTests(unittest.TestCase):
    def test_encode_decode_roundtrip_preserves_float32_shape(self) -> None:
        original = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
        payload = encode_array(original)
        restored = decode_array(payload)
        self.assertEqual(restored.shape, (1, 3, 8))
        self.assertEqual(restored.dtype, np.float32)
        np.testing.assert_allclose(restored, original)

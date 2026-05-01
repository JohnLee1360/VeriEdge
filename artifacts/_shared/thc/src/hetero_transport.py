from __future__ import annotations

import base64
import json
from typing import Any
from urllib import request

import numpy as np


def encode_array(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32, order="C")
    return {
        "shape": list(arr.shape),
        "dtype": "float32",
        "data_b64": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(dim) for dim in payload["shape"])
    dtype = np.dtype(str(payload.get("dtype", "float32")))
    raw = base64.b64decode(str(payload["data_b64"]).encode("ascii"))
    return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32, copy=False)


def post_json(url: str, payload: dict[str, Any], timeout_s: float = 300.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))

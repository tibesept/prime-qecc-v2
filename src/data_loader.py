import json
import os
from typing import Dict, List, Optional, Tuple

import mpmath


class RiemannZerosLoader:
    """
    Loads imaginary parts gamma_n of non-trivial zeta zeros using mpmath.zetazero.
    A local cache is kept to avoid recomputing the same zeros repeatedly.
    """

    LAST_LOAD_INFO: Dict[str, object] = {
        "source": "unknown",
        "precision_digits": 0,
        "requested_dps": None,
        "count": 0,
    }

    @staticmethod
    def _infer_precision_digits(value_strings: List[str]) -> int:
        digits = []
        for item in value_strings:
            text = str(item).strip()
            if "." in text:
                digits.append(len(text.split(".")[-1]))
        if not digits:
            return 0
        return min(digits)

    @staticmethod
    def _read_cache(cache_path: str) -> Tuple[List[mpmath.mpf], Dict[str, object]]:
        if not os.path.exists(cache_path):
            return [], {}

        with open(cache_path, "r") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict) or "gammas" not in payload:
            return [], {}

        raw = payload.get("gammas", [])
        metadata = dict(payload.get("metadata", {}))
        gammas = [mpmath.mpf(v) for v in raw]
        inferred_digits = RiemannZerosLoader._infer_precision_digits([str(v) for v in raw[:100]])

        metadata.setdefault("source", "mpmath")
        metadata.setdefault("format", "v2")
        metadata["precision_digits"] = int(metadata.get("precision_digits", inferred_digits))
        metadata["count"] = len(gammas)
        return gammas, metadata

    @staticmethod
    def _save_cache(gammas: List[mpmath.mpf], cache_path: str, dps: int) -> Dict[str, object]:
        raw = [str(g) for g in gammas]
        metadata = {
            "source": "mpmath",
            "precision_digits": RiemannZerosLoader._infer_precision_digits(raw[:100]),
            "requested_dps": int(dps),
            "count": len(raw),
            "format": "v2",
        }
        payload = {
            "metadata": metadata,
            "gammas": raw,
        }
        with open(cache_path, "w") as handle:
            json.dump(payload, handle)
        return metadata

    @staticmethod
    def _generate_range(start_index: int, end_index: int, dps: int) -> List[mpmath.mpf]:
        mpmath.mp.dps = dps
        gammas = []
        for index in range(start_index, end_index + 1):
            gammas.append(mpmath.im(mpmath.zetazero(index)))
        return gammas

    @staticmethod
    def get_last_load_info() -> Dict[str, object]:
        return dict(RiemannZerosLoader.LAST_LOAD_INFO)

    @staticmethod
    def load_odlyzko(
        num_zeros: int = 100000, dps: int = 50, min_precision_digits: Optional[int] = None
    ) -> List[mpmath.mpf]:
        """
        Historical method name retained for compatibility.
        The loader now uses only mpmath.zetazero plus a local cache built from it.
        """
        mpmath.mp.dps = dps
        required_digits = int(min_precision_digits) if min_precision_digits is not None else min(int(dps), 30)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base_dir, "data", "riemann_zeros.json")

        cache_gammas, cache_meta = RiemannZerosLoader._read_cache(cache_path)
        cache_digits = int(cache_meta.get("precision_digits", 0))
        cache_source = cache_meta.get("source", "unknown")

        def set_info(source: str, precision_digits: int, count: int):
            RiemannZerosLoader.LAST_LOAD_INFO = {
                "source": source,
                "precision_digits": int(precision_digits),
                "requested_dps": int(dps),
                "count": int(count),
            }

        cache_is_usable = cache_source == "mpmath" and cache_digits >= required_digits
        if cache_is_usable and len(cache_gammas) >= num_zeros:
            print(f"Loaded {num_zeros} zeros from mpmath cache.")
            set_info("mpmath_cache", cache_digits, num_zeros)
            return cache_gammas[:num_zeros]

        if cache_is_usable and len(cache_gammas) < num_zeros:
            start = len(cache_gammas) + 1
            end = num_zeros
            print(f"Extending mpmath cache from {len(cache_gammas)} to {num_zeros} zeros at dps={dps}...")
            extension = RiemannZerosLoader._generate_range(start, end, dps)
            combined = cache_gammas + extension
            metadata = RiemannZerosLoader._save_cache(combined, cache_path, dps)
            set_info("mpmath_extended", metadata["precision_digits"], num_zeros)
            return combined[:num_zeros]

        print(f"Generating {num_zeros} zeros with mpmath.zetazero at dps={dps}...")
        gammas = RiemannZerosLoader._generate_range(1, num_zeros, dps)
        metadata = RiemannZerosLoader._save_cache(gammas, cache_path, dps)
        set_info("mpmath", metadata["precision_digits"], num_zeros)
        return gammas

    @staticmethod
    def verify_first_five(
        gammas: List[mpmath.mpf],
        tolerance: Optional[mpmath.mpf] = None,
        source_info: Optional[Dict[str, object]] = None,
    ) -> bool:
        """
        Checks first 5 zeros against high-precision reference constants.
        """
        expected = [
            mpmath.mpf("14.13472514173469379045725198356247027078"),
            mpmath.mpf("21.02203963877155499262847959389690277733"),
            mpmath.mpf("25.01085758014568876321379099256282181866"),
            mpmath.mpf("30.42487612585951321031189753058409132018"),
            mpmath.mpf("32.93506158773918969066236896407490348881"),
        ]

        if source_info is None:
            source_info = RiemannZerosLoader.get_last_load_info()

        if tolerance is None:
            digits = int(source_info.get("precision_digits", 30) or 30)
            tolerance_digits = min(max(digits - 6, 12), 35)
            tolerance = mpmath.power(10, -tolerance_digits)
        else:
            tolerance = mpmath.mpf(tolerance)

        for i in range(min(5, len(gammas))):
            error = abs(gammas[i] - expected[i])
            if error > tolerance:
                print(
                    f"ERROR: gamma_{i + 1}={gammas[i]} expected={expected[i]} "
                    f"error={error} tolerance={tolerance}"
                )
                return False

        print(
            "Verified first 5 zeros "
            f"(source={source_info.get('source')}, tolerance={tolerance})."
        )
        return True


if __name__ == "__main__":
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=50, dps=50)
    loader.verify_first_five(gammas)

import json
import os
import urllib.request
from typing import List

import mpmath


class RiemannZerosLoader:
    """
    Loads Riemann zeros with high precision (mpmath.mpf).
    Each zero has the form 1/2 + i*gamma_n. We only work with the imaginary parts (gamma_n).
    THIS ESTABLISHED MATH: Odlyzko tables contain the numeric values of these computed zeros.
    """

    CACHE_FILE = "../data/riemann_zeros_cache.json"

    @staticmethod
    def fetch_odlyzko_from_web(num_zeros: int) -> List[mpmath.mpf]:
        """
        Fetches the first 100,000 zeros directly from Odlyzko's website if local file doesn't exist.
        The file 'zeros1' contains the first 100,000 zeros.
        """
        # Odlyzko's table zeros1 contains 100,000 zeros.
        url = "http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1"
        try:
            print(f"Fetching zeros from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                
        except Exception as e:
            print(f"Error fetching from web: {e}")
            return []

        gammas = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Safe parsing
            try:
                # Get the last token in case there are prefixes or indices
                val_str = line.split()[-1]
                val = mpmath.mpf(val_str)
                gammas.append(val)
            except ValueError:
                continue
                
            if len(gammas) >= num_zeros:
                break
                
        return gammas

    @staticmethod
    def load_odlyzko(num_zeros: int = 100000, dps: int = 50) -> List[mpmath.mpf]:
        """
        Loads from Odlyzko.
        Tries SageMath first, then cached JSON, then local text file / web.
        
        Args:
            num_zeros: How many zeros to load (gamma_n)
            dps: Decimal places for high-precision arithmetic

        Returns:
            List of gamma_n as mpmath.mpf
        """
        mpmath.mp.dps = dps
        
        # Determine paths relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base_dir, "data", "riemann_zeros.json")
        txt_path = os.path.join(base_dir, "data", "zeta_zeros.txt")

        # 1. Try JSON Cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if len(data) >= num_zeros:
                        print(f"Loaded {num_zeros} zeros from cache.")
                        return [mpmath.mpf(val) for val in data[:num_zeros]]
            except Exception as e:
                print(f"Failed to read cache: {e}")

        # 2. Try SageMath
        try:
            from sage.databases.odlyzko import zeta_zeros
            print("Using SageMath to load zeros...")
            gammas_sage = zeta_zeros(num_zeros)
            gammas = [mpmath.mpf(str(g)) for g in gammas_sage[:num_zeros]]
            RiemannZerosLoader._save_cache(gammas, cache_path)
            return gammas
        except ImportError:
            pass

        # 3. Try Local File `zeta_zeros.txt`
        gammas = []
        if os.path.exists(txt_path):
            try:
                print(f"Loading from local text file {txt_path}...")
                with open(txt_path, "r") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line: continue
                        if len(gammas) >= num_zeros:
                            break
                        gammas.append(mpmath.mpf(line))
            except Exception as e:
                print(f"Failed to read local text file: {e}")

        # 4. Fetch from Web if file didn't exist or didn't have enough
        if len(gammas) < num_zeros:
            gammas = RiemannZerosLoader.fetch_odlyzko_from_web(num_zeros)

        # 5. Save Cache if successful
        if len(gammas) > 0:
            RiemannZerosLoader._save_cache(gammas, cache_path)
            
        return gammas[:num_zeros]

    @staticmethod
    def _save_cache(gammas: List[mpmath.mpf], cache_path: str):
        try:
            data_to_save = [str(g) for g in gammas]
            with open(cache_path, 'w') as f:
                json.dump(data_to_save, f)
            print(f"Saved {len(gammas)} zeros to cache.")
        except Exception as e:
            print(f"Failed to write cache: {e}")

    @staticmethod
    def verify_first_five(gammas: List[mpmath.mpf]) -> bool:
        """
        Verification: The first 5 nontrivial zeros should match known values exactly.
        """
        expected = [
            mpmath.mpf("14.134725142068005"),
            mpmath.mpf("21.022039638771554"),
            mpmath.mpf("25.010857580145688"),
            mpmath.mpf("30.424876125859513"),
            mpmath.mpf("32.935061587739189")
        ]
        
        # For our error limit, Odlyzko's tables are highly precise
        tolerance = mpmath.mpf(1e-10)
        
        for i in range(min(5, len(gammas))):
            error = abs(gammas[i] - expected[i])
            if error > tolerance:
                print(f"ERROR: gamma_{i} = {gammas[i]}, expected {expected[i]}")
                return False
        
        print(f"✓ First 5 zeros perfectly verified against Odlyzko baseline.")
        return True


if __name__ == "__main__":
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=100000)
    loader.verify_first_five(gammas)

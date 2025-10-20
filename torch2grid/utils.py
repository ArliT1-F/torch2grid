import sys
import time
from typing import Iterator, Any, Optional

from numpy import bytes_


def progress_bar(iterable: Iterator[Any],
                 total: Optional[int] = None,
                 desc: str = "Processing",
                 width: int = 50) -> Iterator[Any]:
    

    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    start_time = time.time()

    for i, item in enumerate(iterable):
        yield item

        if total is not None:
            progress = (i + 1) / total
            filled = int(width * progress)
            bar = '[' + '=' * filled + ' ' * (width - filled) + ']'

            elapsed = time.time() - start_time
            if progress > 0:
                eta = elapsed / progress - elapsed
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: ?"

            print(f"\r{desc}: |{bar}| {progress:.1%} ({i+1}/{total}) {eta_str}", end='', flush=True)



    print()


def print_section(title: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)



def format_bytes(bytes_value: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_number(number: float, precision: int = 3) -> str:
    if abs(number) < 1e-6:
        return f"{number:.{precision}e}"
    elif abs(number) < 1e-3:
        return f"{number:.{precision}f}"
    elif abs(number) < 1e6:
        return f"{number:.{precision}f}"
    elif abs(number) < 1e9:
        return f"{number/1e6:.{precision}f}M"
    elif abs(number) < 1e12:
        return f"{number/1e9:.{precision}f}B"
    else:
        return f"{number:.{precision}e}"
    

def safe_filename(filename: str) -> str:
    import re
    
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)

    safe = re.sub(r'_+', '_', safe)

    safe = safe.strip('_.')

    if not safe:
        safe = 'unnamed'
    return safe
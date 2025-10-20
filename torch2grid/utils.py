"""
Utility functions for torch2grid.
"""

import sys
import time
from typing import Iterator, Any, Optional


def progress_bar(iterable: Iterator[Any], 
                total: Optional[int] = None, 
                desc: str = "Processing",
                width: int = 50) -> Iterator[Any]:
    """
    Create a simple progress bar for iterables.
    
    Args:
        iterable: The iterable to wrap
        total: Total number of items (if known)
        desc: Description to show
        width: Width of the progress bar
        
    Yields:
        Items from the iterable
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    
    start_time = time.time()
    
    for i, item in enumerate(iterable):
        yield item
        
        if total is not None:
            # Calculate progress
            progress = (i + 1) / total
            filled = int(width * progress)
            bar = '█' * filled + '░' * (width - filled)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            if progress > 0:
                eta = elapsed / progress - elapsed
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: ?"
            
            # Print progress bar
            print(f"\r{desc}: |{bar}| {progress:.1%} ({i+1}/{total}) {eta_str}", end='', flush=True)
    
    # Print final newline
    print()


def print_section(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the section header
    """
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_number(number: float, precision: int = 3) -> str:
    """
    Format a number with appropriate precision and units.
    
    Args:
        number: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string (e.g., "1.23e-06", "1.23M")
    """
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
    """
    Convert a string to a safe filename.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename with invalid characters replaced
    """
    import re
    # Replace invalid characters with underscores
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Remove leading/trailing underscores and dots
    safe = safe.strip('_.')
    # Ensure it's not empty
    if not safe:
        safe = 'unnamed'
    return safe
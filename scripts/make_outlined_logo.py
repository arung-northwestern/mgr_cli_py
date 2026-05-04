"""
Create a transparent-background logo with black outline from the dark-bg logo.

Usage:
    python scripts/make_outlined_logo.py
    # Output: docs/mofography_logo_outlined.png
"""

from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import numpy as np


def make_outlined_logo(
    input_path: Path,
    output_path: Path,
    stroke_width: int = 4,
    stroke_color: tuple = (0, 0, 0, 255),
    bg_threshold: int = 60,
):
    """
    Convert a logo with dark background to transparent background + black outline.

    Parameters
    ----------
    input_path : Path
        Source PNG (white text on dark background).
    output_path : Path
        Destination PNG (transparent bg, white text, black outline).
    stroke_width : int
        Outline thickness in pixels.
    stroke_color : tuple
        RGBA color for the outline (default black).
    bg_threshold : int
        Brightness threshold below which pixels are treated as background.
    """
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    # Compute luminance: treat dark pixels as background
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Create mask: bright pixels = text, dark pixels = background
    text_mask = luminance > bg_threshold

    # Separate text (white-ish) and colored logo elements (blue-ish)
    # The MOFography logo has white text + blue icon. Keep both.
    # Use saturation or just keep anything not very dark.
    max_channel = np.maximum(np.maximum(r, g), b)
    non_bg_mask = max_channel > bg_threshold

    # Create text-only RGBA image (transparent background)
    rgba = data.copy()
    rgba[~non_bg_mask] = [0, 0, 0, 0]  # Make background transparent

    text_img = Image.fromarray(rgba, mode="RGBA")

    # Create outline by dilating the alpha mask
    alpha = text_img.split()[-1]
    # Dilate = max filter
    outline = alpha.filter(ImageFilter.MaxFilter(stroke_width * 2 + 1))

    # Build outline layer (black stroke)
    outline_rgba = Image.new("RGBA", img.size, stroke_color)
    outline_rgba.putalpha(outline)

    # Composite: outline behind, original text on top
    # Where text is transparent, show outline; where text is opaque, show text
    result = Image.alpha_composite(outline_rgba, text_img)

    result.save(output_path)
    print(f"Saved outlined logo to {output_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    input_logo = repo_root / "docs" / "mofography_logo_dark_background.png"
    output_logo = repo_root / "docs" / "mofography_logo_outlined.png"

    if not input_logo.is_file():
        print(f"Input logo not found: {input_logo}")
        exit(1)

    make_outlined_logo(input_logo, output_logo, stroke_width=4)

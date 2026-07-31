import matplotlib.pyplot as plt


def get_class_colors(classes: list[str]) -> dict[str, tuple[int, int, int]]:
    """Deterministic tab10-based color per class, shared between the API overlay and the Streamlit legend."""
    palette = plt.colormaps["tab10"].resampled(len(classes))
    return {c: tuple(int(v * 255) for v in palette(i)[:3]) for i, c in enumerate(classes)}

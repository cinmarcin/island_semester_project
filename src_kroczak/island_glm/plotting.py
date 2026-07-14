from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_template
from nilearn.image import load_img, resample_to_img
from nilearn.plotting import plot_stat_map, view_img


def plot_group_stat_map(
    z_map_path,
    cut_coords,
    title,
    threshold=2.33,
    vmax=5.0,
    output_path=None,
    bg_img=None,
    dim=-0.2,
):
    """Plot a group-level z-map.

    If bg_img is provided, it is used directly as the anatomical background.
    We intentionally do not resample the anatomical background down to the
    z-map grid, because that makes the T1w image look blocky.
    """

    import matplotlib.pyplot as plt

    from pathlib import Path
    from nilearn.image import load_img
    from nilearn.datasets import load_mni152_template
    from nilearn.plotting import plot_stat_map

    z_img = load_img(str(z_map_path))

    if bg_img is None:
        bg_img = load_mni152_template(resolution=2)

    fig = plt.figure(figsize=(12, 5))

    display = plot_stat_map(
        z_img,
        bg_img=bg_img,
        display_mode="ortho",
        cut_coords=cut_coords,
        threshold=threshold,
        vmax=vmax,
        cmap="cold_hot",
        black_bg=False,
        title=title,
        figure=fig,
        dim=dim,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return display


def make_interactive_view(z_map_path: str | Path, threshold: float = 2.33, title: str | None = None):
    return view_img(str(z_map_path), threshold=threshold, title=title, cmap="cold_hot")

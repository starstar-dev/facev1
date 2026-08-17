import os
import math
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

PAPER_WIDTH_IN = 7.16
PAPER_DPI = 600
PAPER_PDF_DPI = 300
QMAP_CMAP = "turbo"
QMAP_ALPHA = 0.58
QUALITY_FLOOR = 0.15

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "savefig.facecolor": "white",
}


def save_figure_pdf_and_png(fig, save_path):
    """Save vector text/lines plus high-resolution embedded raster panels."""
    stem, _ = os.path.splitext(save_path)
    pdf_path = f"{stem}.pdf"
    png_path = f"{stem}.png"

    parent = os.path.dirname(os.path.abspath(stem))
    os.makedirs(parent, exist_ok=True)

    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
    }
    # Matplotlib otherwise embeds imshow panels in PDF at the default figure
    # DPI (normally 100 ppi), which makes the PDF look softer and paler than
    # the 600-dpi PNG when it is enlarged in a viewer.
    fig.savefig(pdf_path, dpi=PAPER_PDF_DPI, transparent=False, **save_kwargs)
    fig.savefig(png_path, dpi=PAPER_DPI, **save_kwargs)
    return pdf_path, png_path


def blend_qmap_overlay(img_np, bad_map, norm):
    """Pre-composite a heatmap so PDF and PNG viewers render it identically."""
    heat_rgb = plt.get_cmap(QMAP_CMAP)(norm(bad_map))[..., :3]
    return ((1.0 - QMAP_ALPHA) * img_np + QMAP_ALPHA * heat_rgb).clip(0.0, 1.0)


def denorm_clip(img):
    """
    img: [3,H,W], CLIP normalized
    """
    mean = img.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = img.new_tensor(CLIP_STD).view(3, 1, 1)
    img = img * std + mean
    return img.clamp(0, 1)


def save_qmap_overlay(img, q_map, save_path, title="q_map"):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    img = denorm_clip(img).detach().float().cpu()
    img_np = img.permute(1, 2, 0).numpy()

    bad_map = qmap_to_badmap(q_map, size=img_np.shape[:2])

    bad_mean = float(bad_map.mean())
    bad_max = float(bad_map.max())

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(PAPER_WIDTH_IN, 3.1))
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[1, 1, 0.045],
            left=0.02,
            right=0.94,
            bottom=0.05,
            top=0.88,
            wspace=0.06,
        )

        ax_original = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        ax_original.imshow(img_np, interpolation="lanczos")
        ax_original.set_title("Original", pad=3)
        ax_original.axis("off")

        norm = Normalize(vmin=0.0, vmax=1.0)
        overlay = blend_qmap_overlay(img_np, bad_map, norm)
        ax_map.imshow(overlay, interpolation="lanczos")
        im = ScalarMappable(norm=norm, cmap=QMAP_CMAP)
        im.set_array([])
        ax_map.set_title(
            f"{title} (mean={bad_mean:.3f}, max={bad_max:.3f})",
            pad=3,
        )
        ax_map.axis("off")

        cbar = fig.colorbar(im, cax=cax, ticks=[0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=7, length=2, width=0.6)
        cbar.set_label(
            "Normalized degradation response",
            fontsize=8,
            labelpad=4,
        )

        save_figure_pdf_and_png(fig, save_path)
        plt.close(fig)


def save_batch_qmap_visualization(model, save_dir, epoch, max_samples=4):
    """
    Save one paper-style visualization figure:
    each row = RGB image | RGB low-quality map | NI image | NI low-quality map.
    Top-k samples are selected by low-quality score.
    """
    import torch

    raw_model = model.module if hasattr(model, "module") else model

    if not hasattr(raw_model, "_last_q_R_map"):
        return

    os.makedirs(save_dir, exist_ok=True)

    img_R = raw_model._last_img_R
    img_N = raw_model._last_img_N
    q_R = raw_model._last_q_R_map
    q_N = raw_model._last_q_N_map

    # Select samples with stronger degradation response.
    bad_score_R = (1.0 - q_R).mean(dim=(1, 2))
    bad_score_N = (1.0 - q_N).mean(dim=(1, 2))
    bad_score = bad_score_R + bad_score_N

    k = min(max_samples, img_R.size(0))
    topk_idx = torch.topk(bad_score, k=k).indices.tolist()

    save_paper_qmap_grid(
        img_R=img_R,
        img_N=img_N,
        q_R=q_R,
        q_N=q_N,
        indices=topk_idx,
        save_path=os.path.join(save_dir, f"epoch{epoch:03d}_paper_qmap_grid.png")
    )


def qmap_to_badmap(q_map, size=(224, 224)):
    import torch.nn.functional as F

    q = q_map.detach().float().cpu()
    if q.dim() == 2:
        q = q.squeeze(-1)

    # Infer the square patch grid instead of hard-coding CLIP ViT-B/16's 14x14
    # layout. This keeps the visualizer valid if the input/backbone changes.
    num_patches = q.numel()
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"Expected a square patch map, but received {num_patches} values"
        )

    q = q.reshape(1, 1, grid_size, grid_size)
    # The model stores quality in [QUALITY_FLOOR, 1]. Invert that affine
    # transform so the displayed degradation response uses the full [0, 1]
    # color scale for every sample. This is a global calibration, not a
    # per-image min-max normalization.
    bad_map = ((1.0 - q) / (1.0 - QUALITY_FLOOR)).clamp(0.0, 1.0)

    bad_map = F.interpolate(
        bad_map,
        size=size,
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()

    return bad_map


def tensor_to_img_np(img):
    img = denorm_clip(img).detach().float().cpu()
    return img.permute(1, 2, 0).numpy()


def save_paper_qmap_grid(img_R, img_N, q_R, q_N, indices, save_path):
    """
    Paper-style visualization:
    rows are selected samples;
    columns are RGB image, RGB low-quality map, NI image, NI low-quality map.
    """
    if len(indices) == 0:
        raise ValueError("indices must contain at least one sample")

    rgb_images = []
    nir_images = []
    bad_R_maps = []
    bad_N_maps = []

    for i in indices:
        rgb_np = tensor_to_img_np(img_R[i])
        nir_np = tensor_to_img_np(img_N[i])
        rgb_images.append(rgb_np)
        nir_images.append(nir_np)
        bad_R_maps.append(qmap_to_badmap(q_R[i], size=rgb_np.shape[:2]))
        bad_N_maps.append(qmap_to_badmap(q_N[i], size=nir_np.shape[:2]))

    return render_paper_qmap_grid(
        rgb_images=rgb_images,
        nir_images=nir_images,
        bad_R_maps=bad_R_maps,
        bad_N_maps=bad_N_maps,
        save_path=save_path,
    )


def render_paper_qmap_grid(
    rgb_images,
    nir_images,
    bad_R_maps,
    bad_N_maps,
    save_path,
):
    """Render precomputed NumPy images/maps using the final paper layout."""
    num_rows = len(rgb_images)
    num_cols = 4

    lengths = {
        len(rgb_images),
        len(nir_images),
        len(bad_R_maps),
        len(bad_N_maps),
    }
    if lengths != {num_rows} or num_rows == 0:
        raise ValueError("All image/map lists must have the same non-zero length")

    col_titles = [
        "RGB",
        "RGB low-quality map",
        "NI",
        "NI low-quality map"
    ]

    # The height is tied to the final double-column width so square image panels
    # remain large enough while leaving a narrow, dedicated colorbar column.
    paper_height = 0.30 + num_rows * 1.57
    norm = Normalize(vmin=0.0, vmax=1.0)

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(PAPER_WIDTH_IN, paper_height))
        gs = fig.add_gridspec(
            num_rows,
            num_cols + 1,
            width_ratios=[1, 1, 1, 1, 0.045],
            left=0.012,
            right=0.94,
            bottom=0.012,
            top=0.955,
            wspace=0.055,
            hspace=0.055,
        )

        axes = [
            [fig.add_subplot(gs[row, col]) for col in range(num_cols)]
            for row in range(num_rows)
        ]
        cax = fig.add_subplot(gs[:, num_cols])

        for c in range(num_cols):
            axes[0][c].set_title(col_titles[c], pad=3)

        last_im = None

        for row, (rgb_np, nir_np, bad_R, bad_N) in enumerate(zip(
            rgb_images,
            nir_images,
            bad_R_maps,
            bad_N_maps,
        )):

            axes[row][0].imshow(rgb_np, interpolation="lanczos")
            axes[row][0].axis("off")

            rgb_overlay = blend_qmap_overlay(rgb_np, bad_R, norm)
            axes[row][1].imshow(rgb_overlay, interpolation="lanczos")
            axes[row][1].axis("off")

            axes[row][2].imshow(nir_np, interpolation="lanczos")
            axes[row][2].axis("off")

            nir_overlay = blend_qmap_overlay(nir_np, bad_N, norm)
            axes[row][3].imshow(nir_overlay, interpolation="lanczos")
            axes[row][3].axis("off")

        last_im = ScalarMappable(norm=norm, cmap=QMAP_CMAP)
        last_im.set_array([])

        # The colorbar owns a GridSpec column, so it can never cover an image.
        cbar = fig.colorbar(
            last_im,
            cax=cax,
            ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        cbar.ax.tick_params(labelsize=7, length=2, width=0.6)
        cbar.set_label(
            "Normalized degradation response",
            fontsize=8,
            labelpad=4,
        )

        output_paths = save_figure_pdf_and_png(fig, save_path)
        plt.close(fig)
        return output_paths

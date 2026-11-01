"""
Decorrelation stretch for RGB images (drone / camera photos), with
configurable working color space.

Reference: Alley, R.E. (1996), "Algorithm Theoretical Basis Document for
Decorrelation Stretch", v2.2, JPL / ASTER AST06.
https://dstretch.com/DecorrelationStretch.pdf

Pipeline:
  1. Convert the image to a configurable working space (srgb, linear, lab,
     ycbcr) -- optional subset of channels only.
  2. Sample pixels (no cloud/bad-pixel masks needed for ordinary photos;
     fully saturated pixels are optionally excluded from the statistics).
  3. Compute the covariance matrix of the selected channels and build the
     whitening transform T (see `decorrelation_stretch` for the two modes).
  4. out = T @ (x - mean) + target_mean, convert back to sRGB, clip.

The stretch math is identical in every space; only the pixel representation
changes before/after. Use `channels` to restrict the stretch to a subset
of the space's channels (e.g. channels=(1, 2) in LAB = a/b only, holding
lightness, like DStretch's LAB-style mode; channels=(1, 2) in YCbCr =
chroma only, holding luma).
"""

import argparse

import numpy as np
from PIL import Image, ImageFilter

# --------------------------------------------------------------------------
# Color space conversions (float arrays in [0, 1] sRGB <-> working space)
# --------------------------------------------------------------------------

SPACES = ("srgb", "linear", "lab", "ycbcr")

_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ2RGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)
_WHITE = np.array([0.95047, 1.0, 1.08883])  # D65


def _decode_srgb(c):
    """[0,1] gamma-encoded -> linear."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _encode_srgb(c):
    """linear -> [0,1] gamma-encoded (input clipped at 0)."""
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _f_lab(t):
    d = 6.0 / 29.0
    return np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)


def _f_lab_inv(t):
    d = 6.0 / 29.0
    return np.where(t > d, t**3, 3 * d * d * (t - 4.0 / 29.0))


def rgb01_to_space(rgb01, space):
    """(..., 3) float sRGB in [0,1] -> working-space values.

    Scales: srgb/linear/ycbcr map to ~[0,255]; lab uses L*2.55 (so L covers
    [0,255]) with a/b in native CIE units (~[-128, 127]).
    """
    if space == "srgb":
        return rgb01 * 255.0
    if space == "linear":
        return _decode_srgb(np.clip(rgb01, 0.0, 1.0)) * 255.0
    if space == "lab":
        lin = _decode_srgb(np.clip(rgb01, 0.0, 1.0))
        xyz = lin @ _RGB2XYZ.T / _WHITE
        fx, fy, fz = (_f_lab(xyz[..., i]) for i in range(3))
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return np.stack([L * 2.55, a, b], axis=-1)
    if space == "ycbcr":
        r, g, b = (rgb01[..., i] * 255.0 for i in range(3))  # BT.601 full-range
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return np.stack([y, cb, cr], axis=-1)
    raise ValueError(f"unknown space {space!r}; expected one of {SPACES}")


def space_to_rgb01(vals, space):
    """Working-space values -> (..., 3) float sRGB.

    sRGB/linear results are NOT clipped to [0,1]: out-of-range values are
    handled later by the adaptive percentile remap. LAB/YCbCr results are
    clipped, because out-of-gamut colors have no sRGB equivalent.
    """
    if space == "srgb":
        return vals / 255.0
    if space == "linear":
        return _encode_srgb(np.maximum(vals, 0.0) / 255.0)
    if space == "lab":
        L, a, b = vals[..., 0] / 2.55, vals[..., 1], vals[..., 2]
        fy = (L + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0
        xyz = (
            np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=-1) * _WHITE
        )
        lin = xyz @ _XYZ2RGB.T
        return np.clip(_encode_srgb(lin), 0.0, 1.0)
    if space == "ycbcr":
        y, cb, cr = vals[..., 0], vals[..., 1] - 128.0, vals[..., 2] - 128.0
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return np.clip(np.stack([r, g, b], axis=-1) / 255.0, 0.0, 1.0)
    raise ValueError(f"unknown space {space!r}; expected one of {SPACES}")


# --------------------------------------------------------------------------
# Decorrelation stretch
# --------------------------------------------------------------------------


def _stats_filter(X, exclude_saturated, margin):
    """X: (n, 3) uint8-range RGB sample."""
    n_total = X.shape[0]
    if exclude_saturated:
        sat = ((X <= margin) | (X >= 255 - margin)).any(axis=1)
        X = X[~sat]
        if X.shape[0] < max(1000, 0.1 * n_total):
            return X if X.shape[0] else None
    return X


def decorrelation_stretch(
    img,
    space="srgb",
    channels=None,
    mode="equal_variance",
    target_std=None,
    target_mean="input",
    clip_fraction=0.004,
    exclude_saturated=True,
    saturation_margin=2,
    sample_stride=None,
):
    """
    img              : HxWx3 uint8 RGB (or path to an image file).
    space            : working space -- "srgb", "linear", "lab" or "ycbcr".
    channels         : indices (within the space) to stretch; others pass
                       through unchanged. Default: all. Examples:
                       ("lab", (1, 2))   -> a/b stretch, lightness preserved;
                       ("ycbcr", (1, 2)) -> chroma stretch, luma preserved.
    mode             : "equal_variance" (default) -- output channels get std
                       exactly target_std (ASTER covariance option,
                       generalized); or "equal_weight" -- output std
                       proportional to each channel's input std (ASTER
                       correlation option, made self-consistent; use this
                       to avoid one high-variance channel dominating).
    target_std       : target output std per stretched channel: a scalar, a
                       per-channel sequence, or None for per-space defaults
                       (srgb/linear: 50 like the ASTER doc; lab: 35/20/20;
                       ycbcr: 50/18/18 -- chroma channels need much less,
                       or most pixels land out of gamut).
    target_mean      : "input" (preserve the mean; default), "byte" (127.5),
                       or a per-channel sequence in the working space.
    clip_fraction    : fraction of pixels per RGB channel allowed to
                       saturate at 0/255 (DStretch-like), applied only for
                       full-channel stretches in display spaces (srgb,
                       linear). All other modes gamut-clip only, so the
                       transform's output is preserved.
    exclude_saturated: drop pure black/white pixels from the statistics.
    """
    if space not in SPACES:
        raise ValueError(f"unknown space {space!r}; expected one of {SPACES}")
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    H, W, _ = arr.shape
    ch = tuple(channels) if channels is not None else (0, 1, 2)
    if not ch or any(not (0 <= c < 3) for c in ch) or len(set(ch)) != len(ch):
        raise ValueError(f"invalid channels {channels!r}")
    k = len(ch)

    # --- 1. stats sampling (every 3rd row/col per doc; all pixels if too few)
    if sample_stride is None:
        sample_stride = 3 if H > 300 or W > 300 else 1
    sample = arr[::sample_stride, ::sample_stride].reshape(-1, 3)
    if sample.shape[0] < 1000:
        sample = arr.reshape(-1, 3)
    sample = _stats_filter(
        sample.astype(np.float64), exclude_saturated, saturation_margin
    )
    if sample is None or sample.shape[0] < 1000:
        raise ValueError("Fewer than 1000 usable pixels (doc threshold)")

    # --- 2. channel stats in the working space
    D = rgb01_to_space(sample / 255.0, space)[:, ch]
    mean = D.mean(axis=0)
    cov = np.atleast_2d(np.cov(D, rowvar=False))
    d_std = np.sqrt(np.diag(cov))
    if (d_std <= 0).any():
        raise ValueError("A selected channel has zero variance")

    # --- 3/4/5. transform. Two modes; both are pure matrix functions of the
    # statistics (well-defined even when eigenvalues are nearly degenerate):
    #   equal_variance: T = R diag(target/sqrt(lam)) R^T with R, lam from the
    #     COVARIANCE matrix -- symmetric whitening. Output channels are
    #     uncorrelated with std exactly = target (before display clipping).
    #     This is the ASTER ATBD covariance option, and the only version of
    #     the correlation option that stays self-consistent when channel
    #     variances differ.
    #   equal_weight: the ATBD correlation option, done consistently:
    #     decorrelate the z-scored channels (Corr^-1/2), conjugate back to
    #     raw units. Output channels are uncorrelated with std proportional
    #     to each channel's input std (equal weighting per channel), rms
    #     output std = target.
    _STD_DEFAULTS = {
        "srgb": (50.0, 50.0, 50.0),
        "linear": (50.0, 50.0, 50.0),
        "lab": (35.0, 20.0, 20.0),
        "ycbcr": (50.0, 18.0, 18.0),
    }
    if target_std is None:
        s_target = np.array([_STD_DEFAULTS[space][c] for c in ch], dtype=np.float64)
    else:
        s_target = np.broadcast_to(np.asarray(target_std, np.float64), (k,))
    if mode == "equal_variance":
        lam, R = np.linalg.eigh(cov)
        lam = np.maximum(lam, 1e-12)
        T = R @ np.diag(s_target / np.sqrt(lam)) @ R.T
    elif mode == "equal_weight":
        corr = cov / np.outer(d_std, d_std)
        lam, R = np.linalg.eigh(corr)
        lam = np.maximum(lam, 1e-12)
        rms = np.sqrt((d_std**2).mean())
        C = R @ np.diag(s_target / (rms * np.sqrt(lam))) @ R.T  # z-space
        T = d_std[:, None] * C / d_std[None, :]  # D C D^-1
    else:
        raise ValueError(f"invalid mode {mode!r}")
    T = T.astype(np.float32)

    if isinstance(target_mean, str):
        if target_mean == "input":
            tmean = mean.astype(np.float32)
        elif target_mean == "byte":
            tmean = np.full(k, 127.5, dtype=np.float32)
        else:
            raise ValueError(f"invalid target_mean {target_mean!r}")
    else:
        tmean = np.broadcast_to(np.asarray(target_mean, np.float32), (k,))

    ch_idx = np.array(ch)

    # --- 6. apply in chunks: RGB -> space -> stretch
    Wsp = np.empty((H, W, 3), dtype=np.float32)
    for i in range(0, H, 64):
        j = min(i + 64, H)
        v = rgb01_to_space(arr[i:j].astype(np.float32) / 255.0, space)
        v[..., ch_idx] = ((v[..., ch_idx] - mean) @ T.T) + tmean
        Wsp[i:j] = v

    # back to RGB (out-of-gamut values clip here)
    out = np.empty_like(Wsp)
    for i in range(0, H, 64):
        j = min(i + 64, H)
        out[i:j] = space_to_rgb01(Wsp[i:j], space) * 255.0

    # Post-processing. Full-channel modes in display spaces (srgb/linear):
    # DStretch-like adaptive percentile remap per RGB channel. All other
    # modes (partial channels, or non-display spaces like lab/ycbcr where a
    # per-RGB-channel remap would re-correlate the working channels):
    # gamut clip only, so the transform's output is delivered as-is.
    if k == 3 and space in ("srgb", "linear"):
        result = np.empty_like(out)
        for c in range(3):
            flat = out[:, :, c].ravel()
            if clip_fraction > 0:
                lo, hi = np.quantile(flat, [clip_fraction, 1.0 - clip_fraction])
            else:
                lo, hi = flat.min(), flat.max()
            if hi > lo:
                flat = (flat - lo) * (255.0 / (hi - lo))
            result[:, :, c] = flat.reshape(H, W)
    else:
        result = out
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def synthetic_scene(size=256, seed=7):
    """Synthetic, brightness-dominated aerial scene for the self-test."""
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]

    def field(sigma):
        noise = (rng.random((h, w)) * 255).astype(np.uint8)
        blur = Image.fromarray(noise, "L").filter(ImageFilter.GaussianBlur(sigma))
        f = np.asarray(blur, dtype=np.float64)
        return (f - f.mean()) / f.std()

    f1, f2 = field(25), field(40)
    S = np.stack(
        [
            np.array([8, 18, -12]),  # vegetation
            np.array([14, 6, -10]),  # soil
            np.array([-7, -3, 12]),  # water
            np.array([2, 0, -2]),
        ]
    )  # road
    sel = np.select(
        [f1 > 0.4, (f1 < -0.6) & (f2 > 0), f2 < -0.5, np.ones((h, w), bool)],
        [0, 2, 1, 3],
    ).astype(int)
    img = (
        (128 + 40 * field(15) + 12 * f1)[..., None]
        + S[sel]
        + rng.normal(0, 3, (h, w, 3))
    )
    patch = (xx - 155) ** 2 + (yy - 105) ** 2 < 20**2
    img[patch] += np.array([9, -3, -6])
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Decorrelation stretch (AST06) " "with configurable color space"
    )
    p.add_argument("input", nargs="?", help="input image (omit: self-test)")
    p.add_argument("output", nargs="?", help="output image path")
    p.add_argument(
        "--space",
        choices=SPACES,
        default="srgb",
        help="working color space (default: srgb)",
    )
    p.add_argument(
        "--channels",
        default=None,
        help="comma-separated channel indices to stretch, "
        "e.g. '1,2' for LAB a/b or YCbCr chroma only",
    )
    p.add_argument(
        "--equal-weight",
        action="store_true",
        help="equal-weight mode (ASTER correlation option): "
        "output std proportional to input channel std",
    )
    p.add_argument(
        "--std",
        type=str,
        default=None,
        help="target std, scalar or comma list (default: "
        "50 for srgb/linear; 35/20/20 for lab; 50/12/12 "
        "for ycbcr)",
    )
    p.add_argument(
        "--clip",
        type=float,
        default=0.004,
        help="fraction of pixels clipped per channel end",
    )
    a = p.parse_args()

    if not a.input:
        # --- self-test on a synthetic, brightness-dominated aerial scene
        src = Image.fromarray(synthetic_scene())
    else:
        src = Image.open(a.input).convert("RGB")

    ch = tuple(int(c) for c in a.channels.split(",")) if a.channels else None
    if a.std is None:
        std = None
    elif "," in a.std:
        std = [float(v) for v in a.std.split(",")]
    else:
        std = float(a.std)
    out = decorrelation_stretch(
        src,
        space=a.space,
        channels=ch,
        mode="equal_weight" if a.equal_weight else "equal_variance",
        target_std=std,
        clip_fraction=a.clip,
    )

    # report decorrelation achieved, measured in the working space
    arr_in = np.asarray(src).reshape(-1, 3).astype(np.float64)
    arr_out = np.asarray(out).reshape(-1, 3).astype(np.float64)
    ch_idx = np.array(ch if ch else (0, 1, 2))
    Din = rgb01_to_space(arr_in / 255.0, a.space)[:, ch_idx]
    Dout = rgb01_to_space(arr_out / 255.0, a.space)[:, ch_idx]
    cin = np.corrcoef(Din, rowvar=False)
    cout = np.corrcoef(Dout, rowvar=False)
    off = np.triu_indices(len(ch_idx), 1)
    print(f"space={a.space} channels={list(ch_idx)}")
    print("input  channel correlations:", np.round(cin[off], 3))
    print("output channel correlations:", np.round(cout[off], 3))
    print("output std/mean:", np.round(Dout.std(0), 1), np.round(Dout.mean(0), 1))

    out.save(a.output or "dstretched.png")
    print("saved ->", a.output or "dstretched.png")

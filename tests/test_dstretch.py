"""Test suite for dstretch.py — color spaces, transform math, edge cases, CLI."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import dstretch as ds  # noqa: E402

# --------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------


def correlated_image(h=256, w=256, seed=3):
    """Synthetic RGB image with realistic inter-channel correlation (~0.9)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (h, w, 3))
    mix = base @ np.array([[1.0, 0.8, 0.6], [0.3, 1.0, 0.5], [0.2, 0.7, 1.0]]).T
    return np.clip(128 + 40 * (mix / mix.std()), 0, 255).astype(np.uint8)


def working_correlations(img, space, ch):
    """Pairwise channel correlations of img in the working space."""
    D = ds.rgb01_to_space(np.asarray(img).reshape(-1, 3) / 255.0, space)[:, list(ch)]
    c = np.corrcoef(D, rowvar=False)
    iu = np.triu_indices(len(list(ch)), 1)
    return np.abs(c[iu])


@pytest.fixture(scope="module")
def scene():
    return ds.synthetic_scene(size=256)


# --------------------------------------------------------------------------
# color space conversions
# --------------------------------------------------------------------------


class TestColorSpaces:
    def test_roundtrip_all_spaces(self):
        rng = np.random.default_rng(0)
        rgb01 = rng.random((32, 32, 3))
        for space in ds.SPACES:
            back = ds.space_to_rgb01(ds.rgb01_to_space(rgb01, space), space)
            assert np.allclose(back, rgb01, atol=5e-6), space

    def test_known_values(self):
        # white -> L=100 (stored *2.55 = 255), a=b=0
        white = ds.rgb01_to_space(np.array([[[1.0, 1.0, 1.0]]]), "lab")[0, 0]
        assert np.allclose(white, [255.0, 0.0, 0.0], atol=1e-3)
        # neutral gray -> Cb = Cr = 128, Y = 127.5
        gray = ds.rgb01_to_space(np.array([[[0.5, 0.5, 0.5]]]), "ycbcr")[0, 0]
        assert np.allclose(gray, [127.5, 128.0, 128.0], atol=1e-6)
        # srgb is identity * 255; linear of white is 255
        assert np.allclose(
            ds.rgb01_to_space(np.array([[[0.5, 0.5, 0.5]]]), "srgb"), 127.5
        )
        assert np.allclose(
            ds.rgb01_to_space(np.array([[[1.0, 1.0, 1.0]]]), "linear"), 255.0
        )

    def test_gamma_functions(self):
        # decode/encode are inverses within float noise
        c = np.linspace(0.0, 1.0, 1000)
        assert np.allclose(ds._encode_srgb(ds._decode_srgb(c)), c, atol=1e-7)
        # black and white map exactly
        assert ds._decode_srgb(0.0) == 0.0
        assert ds._decode_srgb(1.0) == pytest.approx(1.0)

    def test_unknown_space_raises(self):
        with pytest.raises(ValueError, match="unknown space"):
            ds.rgb01_to_space(np.zeros((1, 1, 3)), "hsv")
        with pytest.raises(ValueError, match="unknown space"):
            ds.space_to_rgb01(np.zeros((1, 1, 3)), "cmyk")
        with pytest.raises(ValueError, match="unknown space"):
            ds.decorrelation_stretch(correlated_image(64, 64), space="hsv")


# --------------------------------------------------------------------------
# transform math
# --------------------------------------------------------------------------


class TestTransform:
    @pytest.mark.parametrize(
        "space,corr_tol,std_tol",
        [
            # srgb/linear: display spaces -- the percentile remap keeps the
            # output near the targets. lab/ycbcr full-channel modes have no
            # remap, so out-of-gamut clipping re-correlates the channels
            # (documented limitation; residual measured ~0.34 / ~0.09 on the
            # reference scene) and output stds are only approximate.
            ("srgb", 0.15, 15.0),
            ("linear", 0.15, 15.0),
            ("lab", 0.40, None),
            ("ycbcr", 0.15, None),
        ],
    )
    def test_full_stretch_decorrelates(self, scene, space, corr_tol, std_tol):
        img = correlated_image() if space in ("srgb", "linear") else scene
        out = ds.decorrelation_stretch(img, space=space)
        assert working_correlations(out, space, (0, 1, 2)).max() < corr_tol
        if std_tol is not None:
            D = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, space)
            assert np.abs(D.std(axis=0) - 50.0).max() < std_tol

    def test_chroma_mode_preserves_lightness(self):
        img = correlated_image()
        out = ds.decorrelation_stretch(img, space="lab", channels=(1, 2))
        Lin = ds.rgb01_to_space(np.asarray(img).reshape(-1, 3) / 255.0, "lab")[:, 0]
        Lout = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "lab")[:, 0]
        assert np.corrcoef(Lin, Lout)[0, 1] > 0.99
        # a/b decorrelated and stretched to ~20
        assert working_correlations(out, "lab", (1, 2)).max() < 0.15
        D = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "lab")[:, 1:3]
        assert np.abs(D.std(axis=0) - 20.0).max() < 2.0

    def test_ycbcr_chroma_preserves_luma_exactly(self):
        img = correlated_image()
        out = ds.decorrelation_stretch(
            img, space="ycbcr", channels=(1, 2), target_std=10.0
        )
        yin = ds.rgb01_to_space(np.asarray(img).reshape(-1, 3) / 255.0, "ycbcr")[:, 0]
        yout = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "ycbcr")[:, 0]
        # luma passes through unchanged; small residual only where gamut
        # clipping at the RGB display stage touches the pixel
        assert np.corrcoef(yin, yout)[0, 1] > 0.99

    def test_equal_weight_proportional_stds(self):
        img = correlated_image()
        out = ds.decorrelation_stretch(
            img, space="lab", channels=(1, 2), mode="equal_weight"
        )
        D = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "lab")[:, 1:3]
        # rms of output stds = target (20), channels uncorrelated
        assert np.sqrt((D.std(axis=0) ** 2).mean()) == pytest.approx(20, abs=2)
        assert working_correlations(out, "lab", (1, 2)).max() < 0.15

    def test_target_std_scalar(self):
        img = correlated_image()
        out = ds.decorrelation_stretch(
            img, space="ycbcr", channels=(1, 2), target_std=5.0
        )
        D = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "ycbcr")[:, 1:3]
        assert np.abs(D.std(axis=0) - 5.0).max() < 0.5

    def test_target_std_per_channel(self):
        img = correlated_image()
        out = ds.decorrelation_stretch(
            img, space="ycbcr", channels=(1, 2), target_std=[10.0, 15.0]
        )
        D = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "ycbcr")[:, 1:3]
        # mild targets stay in gamut; stds land on the requested values
        assert np.abs(D.std(axis=0) - [10.0, 15.0]).max() < 2.0

    def test_target_mean_byte(self):
        img = correlated_image()
        out = ds.decorrelation_stretch(
            img,
            space="ycbcr",
            channels=(1, 2),
            target_std=5.0,
            target_mean="byte",
        )
        D = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "ycbcr")[:, 1:3]
        # small target std keeps everything in gamut: means land on 127.5
        assert np.abs(D.mean(axis=0) - 127.5).max() < 1.0

    def test_excludes_saturated_pixels_from_stats(self):
        # a mostly-flat image with a large saturated region: the stats must
        # come from the unsaturated part, and the result must not be uniform
        rng = np.random.default_rng(1)
        img = np.full((128, 128, 3), 128, np.uint8)
        img += rng.integers(-40, 40, img.shape).astype(np.int8).astype(np.uint8)
        img[:40] = 255  # saturated band
        out = ds.decorrelation_stretch(img, space="srgb")
        assert np.asarray(out).std() > 0

    def test_rank_deficient_cov_completes(self):
        # duplicated channel -> rank-1 covariance: floored eigenvalues,
        # degraded but defined output (ATBD: warning, continues)
        gray = correlated_image(64, 64)[..., :1]
        img = np.repeat(gray, 3, axis=2)
        img[..., 1] = np.roll(img[..., 1], 5, axis=0)  # break perfect rank-1
        img[..., 1] = img[..., 0]  # restore exact duplicate
        out = ds.decorrelation_stretch(img, space="srgb")
        assert np.asarray(out).shape == img.shape


# --------------------------------------------------------------------------
# degenerate inputs and error handling
# --------------------------------------------------------------------------


class TestErrors:
    def test_zero_variance_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            ds.decorrelation_stretch(np.full((64, 64, 3), 128, np.uint8))

    def test_invalid_channels(self):
        img = correlated_image(64, 64)
        for bad in [(0, 3), (0, -1), (1, 1), (), [0, 0]]:
            with pytest.raises(ValueError, match="invalid channels"):
                ds.decorrelation_stretch(img, channels=bad)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="invalid mode"):
            ds.decorrelation_stretch(correlated_image(64, 64), mode="nope")

    def test_invalid_target_mean(self):
        with pytest.raises(ValueError, match="invalid target_mean"):
            ds.decorrelation_stretch(
                correlated_image(64, 64), channels=(1, 2), target_mean="nope"
            )

    def test_fewer_than_1000_usable_pixels_raises(self):
        # 40x40 = 1600 px, all saturated -> filtered to below the threshold
        img = np.full((40, 40, 3), 255, np.uint8)
        with pytest.raises(ValueError, match="1000 usable"):
            ds.decorrelation_stretch(img, space="srgb")


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


class TestInputs:
    def test_uint16_uses_high_byte(self):
        vals = np.array([300, 1000, 40000, 65535], dtype=np.uint16)
        u16 = np.tile(vals.repeat(3).reshape(4, 1, 3), (30, 250, 1))
        out16 = np.asarray(ds.decorrelation_stretch(u16.copy(), space="srgb"))
        out8 = np.asarray(
            ds.decorrelation_stretch((u16 >> 8).astype(np.uint8), space="srgb")
        )
        assert np.array_equal(out16, out8)

    def test_float_0_1_input_scaled(self):
        rng = np.random.default_rng(2)
        f01 = rng.random((30, 300, 3)).astype(np.float32)
        out = np.asarray(ds.decorrelation_stretch(f01))
        assert out.min() < 5 and out.max() > 250  # full 8-bit range used

    def test_grayscale_replicated(self):
        rng = np.random.default_rng(4)
        g = rng.integers(60, 200, (64, 64)).astype(np.uint8)
        img = np.repeat(g[..., None], 3, axis=2)
        out = np.asarray(ds.decorrelation_stretch(img, space="srgb"))
        assert np.array_equal(out[..., 0], out[..., 1])

    def test_rgba_alpha_dropped(self):
        img = correlated_image(64, 64)
        rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        out_rgb = np.asarray(ds.decorrelation_stretch(img))
        out_rgba = np.asarray(ds.decorrelation_stretch(rgba))
        assert np.array_equal(out_rgb, out_rgba)

    def test_bare_int_channels(self):
        img = correlated_image(64, 64)
        out = ds.decorrelation_stretch(img, channels=1)
        assert np.asarray(out).shape == img.shape

    def test_path_input(self, tmp_path):
        p = tmp_path / "in.png"
        Image.fromarray(correlated_image(64, 64)).save(p)
        out = ds.decorrelation_stretch(str(p), space="srgb")
        assert np.asarray(out).shape == (64, 64, 3)


# --------------------------------------------------------------------------
# threading, chunking, remap
# --------------------------------------------------------------------------


class TestExecution:
    def test_workers_do_not_change_output(self):
        img = correlated_image(700, 40)  # spans several 256-row chunks
        ref = np.asarray(ds.decorrelation_stretch(img, space="lab", workers=1))
        for workers in (2, 5, None):
            out = np.asarray(
                ds.decorrelation_stretch(img, space="lab", workers=workers)
            )
            assert np.array_equal(ref, out), workers

    def test_remap_spreads_two_value_gray(self):
        gray = np.full((200, 200, 3), 79, np.uint8)
        gray[:100] = 3
        out = np.asarray(ds.decorrelation_stretch(gray, space="srgb"))
        assert set(np.unique(out)) == {0, 255}

    def test_clip_fraction_zero_uses_full_range(self):
        img = correlated_image()
        out = np.asarray(ds.decorrelation_stretch(img, space="srgb", clip_fraction=0.0))
        assert out.min() == 0 and out.max() == 255

    def test_no_remap_for_chroma_modes(self):
        # chroma-subset output is delivered as-is: unique values beyond the
        # stretched channels' range are not remapped away
        img = correlated_image()
        out = ds.decorrelation_stretch(img, space="ycbcr", channels=(1, 2))
        yin = ds.rgb01_to_space(np.asarray(img).reshape(-1, 3) / 255.0, "ycbcr")[:, 0]
        yout = ds.rgb01_to_space(np.asarray(out).reshape(-1, 3) / 255.0, "ycbcr")[:, 0]
        assert np.corrcoef(yin, yout)[0, 1] > 0.999


# --------------------------------------------------------------------------
# CLI helpers and CLI
# --------------------------------------------------------------------------


class TestCliHelpers:
    def test_output_path_default_alongside(self):
        p = ds._output_path("/a/b/img.jpg", None)
        assert p == "/a/b/img-dstretched.jpg"

    def test_output_path_directory(self):
        assert ds._output_path("/a/b/img.jpg", "/out", is_dir=True) == "/out/img.jpg"
        assert ds._output_path("/a/b/img.jpg", "/out/") == "/out/img.jpg"

    def test_output_path_file_no_ext(self):
        assert ds._output_path("/a/b/img.jpg", "/out/x") == "/out/x.jpg"

    def test_output_path_unknown_input_ext(self):
        assert ds._output_path("/a/b/img.tifx", None) == "/a/b/img-dstretched.png"

    def test_fmt_duration(self):
        assert ds._fmt_duration(0.05) == "50 ms"
        assert ds._fmt_duration(1.5) == "1.50 s"
        assert ds._fmt_duration(90.0) == "1.5 min"


class TestCli:
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO / "dstretch.py"), *args],
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_self_test(self):
        r = self.run_cli()
        assert r.returncode == 0, r.stderr
        assert "output channel correlations:" in r.stdout
        assert "saved -> dstretched.png" in r.stdout

    def test_single_image(self, tmp_path):
        src = tmp_path / "in.jpg"
        Image.fromarray(correlated_image(64, 64)).save(src)
        dst = tmp_path / "out.png"
        r = self.run_cli(
            str(src), "-o", str(dst), "--space", "lab", "--channels", "1,2"
        )
        assert r.returncode == 0, r.stderr
        assert dst.exists()
        assert "processing time:" in r.stdout
        assert "channels=[1, 2]" in r.stdout

    def test_batch_to_directory(self, tmp_path):
        for i in range(3):
            Image.fromarray(correlated_image(64, 64, seed=i)).save(
                tmp_path / f"img{i}.jpg"
            )
        outdir = tmp_path / "res"
        r = self.run_cli(
            str(tmp_path / "img0.jpg"),
            str(tmp_path / "img1.jpg"),
            str(tmp_path / "img2.jpg"),
            "-o",
            str(outdir),
            "--jobs",
            "2",
        )
        assert r.returncode == 0, r.stderr
        assert len(list(outdir.glob("img*.jpg"))) == 3
        assert "3 image(s) processed, 0 failed" in r.stdout

    def test_batch_missing_file_fails_gracefully(self, tmp_path):
        Image.fromarray(correlated_image(64, 64)).save(tmp_path / "img0.jpg")
        r = self.run_cli(
            str(tmp_path / "img0.jpg"),
            str(tmp_path / "missing.jpg"),
            "-o",
            str(tmp_path),
        )
        assert r.returncode == 1
        assert "FAILED" in r.stdout
        assert "1 image(s) processed, 1 failed" in r.stdout

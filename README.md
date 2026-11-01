# Decorrelation stretch

Decorrelation stretch for RGB images (drone / camera photos), with a
configurable working color space — the classic "DStretch look": dramatic
colour separation for aerial imagery, rock art, and pigment analysis.

Reference: Alley, R.E. (1996), *Algorithm Theoretical Basis Document for
Decorrelation Stretch*, v2.2, JPL / ASTER AST06.
<https://dstretch.com/DecorrelationStretch.pdf>

> **Independent implementation.** This is an original, independent
> implementation of a published public-domain algorithm (Taylor 1973;
> Soha & Schwartz 1978; Alley 1996). It is not affiliated with, derived
> from, or endorsed by the author of the DStretch ImageJ plugin. Credit
> where due: the DStretch plugin (J. Harman) popularized this technique
> in rock-art documentation.

## How it works

1. Convert the image to a working color space (`srgb`, `linear`, `lab`,
   `ycbcr`) — optionally a subset of channels only.
2. Sample pixels (no cloud/bad-pixel masks needed for ordinary photos;
   fully saturated pixels are excluded from the statistics).
3. Compute the covariance matrix of the selected channels and build a
   whitening transform `T` with `T Σ Tᵀ = target_std² · I`.
4. `out = T(x − μ) + m`, convert back to sRGB, clip.

Two weighting modes:

- **equal-variance** (default): output channels are uncorrelated with std
  exactly `target_std` (ASTER covariance option, generalized).
- **equal-weight**: the ATBD correlation option, made self-consistent —
  output std proportional to each channel's input std. Use when one
  high-variance channel (CMYK K, Lab L) would otherwise dominate.

It departs from the ATBD in two documented places where that document is
internally inconsistent: the correlation-matrix option with unequal channel
variances, and mean repositioning (default is mean-preserving).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# single image, full report
python dstretch.py photo.jpg -o out.png --space lab --channels 1,2

# batch: a whole session, in parallel
python dstretch.py session/*.JPG -o dstretched/ --jobs 6
```

With no input, the script runs a self-test on a synthetic aerial scene and
reports input/output channel correlations and stds.

### Channel indices per working space

| Space | 0 | 1 | 2 |
|---|---|---|---|
| `srgb` / `linear` | R | G | B |
| `lab` | L* (lightness) | a* (green–red) | b* (blue–yellow) |
| `ycbcr` | Y (luma) | Cb (blue chroma) | Cr (red chroma) |

Unlisted channels pass through unchanged. The useful case is
`--channels 1,2` with `--space lab` or `ycbcr`: lightness/luma is preserved
and only the color information is decorrelated and stretched, so brightness
structure is kept while colors are expanded apart — the LAB-style mode of
the DStretch plugin, useful for faint pigment and vegetation/soil contrasts.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--space` | `srgb` | working color space (`srgb`, `linear`, `lab`, `ycbcr`) |
| `--channels` | all | comma-separated indices to stretch, e.g. `1,2` |
| `--equal-weight` | off | equal-weight mode (see above) |
| `--std` | per space | target std per stretched channel: `50` for srgb/linear, `35/20/20` for lab, `50/18/18` for ycbcr |
| `--clip` | `0.004` | fraction of pixels clipped per channel end (full-channel srgb/linear remap only) |
| `--jobs` | cpu count | images processed concurrently in batch mode |

`python dstretch.py --help` has the full descriptions.

## Performance

The pixel transform is pointwise and runs chunked in a thread pool; the
percentile remap and the quality report are computed on subsamples. On an
M-series MacBook: ~300 ms for a 24 MP image in sRGB, ~600 ms in Lab
(40–70 MP/s); batch mode overlaps JPEG I/O across images and scales with
the core count.

## Known limitations

- **Noise amplification.** The whitening boost lands on the
  lowest-variance principal direction; if that is mostly sensor noise, noise
  is amplified too. `--std` is the tuning knob.
- **Gamut clipping.** Aggressive stretches push pixels out of sRGB gamut;
  they clip at the display stage.
- **Scene-dependent colors.** Statistics are per image; the same material
  maps to different colors in different photos.
- **Bimodal scenes** (dominant uniform water/snow/vegetation) weaken the
  enhancement — the ATBD's own limitation.

## License

[MIT](LICENSE). The algorithm attribution above is a citation of a public
reference, not a claim of derivation.

//! Inverses of the `gemm_sme_*_packb` weight-panel layouts. Reached only on
//! the OOM fallback, so they carry their own round-trip tests.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

/// Unpack an SME-packed 16-bit B buffer (the `gemm_sme_*_packb` tile-major
/// layout, shared by f16f16 and b16b16) back to a row-major `k x n` buffer of
/// `u16` bit patterns. Tile `t` holds `data[t*k*32 + d*32 + j] == B[d, t*32+j]`
/// (columns past `n` are zero-padding). Used only on the OOM fallback path.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn unpack_b16_sme(data: &[u16], n: usize, k: usize) -> Vec<u16> {
    let mut b = vec![0u16; k * n];
    for d in 0..k {
        for j in 0..n {
            let t = j / 32;
            b[d * n + j] = data[t * k * 32 + d * 32 + (j % 32)];
        }
    }
    b
}

/// Unpack an SME-packed f32/f64 B buffer (the `gemm_sme_{f32,f64}_packb`
/// `[2*n_tiles][k,LANES]` band layout) back to a row-major `k x n` buffer.
/// Column `j` lives in band `j / LANES` at lane `j % LANES`. OOM fallback only.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn unpack_b_band<T: Copy + Default>(data: &[T], n: usize, k: usize, lanes: usize) -> Vec<T> {
    let per_tile = k * lanes;
    let mut b = vec![T::default(); k * n];
    for d in 0..k {
        for j in 0..n {
            b[d * n + j] = data[(j / lanes) * per_tile + d * lanes + (j % lanes)];
        }
    }
    b
}

/// f32 band-panel unpack (16 lanes per band). See [`unpack_b_band`].
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn unpack_b_f32_sme(data: &[f32], n: usize, k: usize) -> Vec<f32> {
    unpack_b_band(data, n, k, 16)
}

/// f64 band-panel unpack (8 lanes per band). See [`unpack_b_band`].
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn unpack_b_f64_sme(data: &[f64], n: usize, k: usize) -> Vec<f64> {
    unpack_b_band(data, n, k, 8)
}

/// Unpack an SME-packed i16 B buffer (`gemm_sme_i16i64_packb`'s
/// `[2*n_tiles][ceil(k/4), 32]` 4-way-interleaved 8-wide bands) back to a
/// row-major `k x n` buffer:
/// `b_pack[band*per_tile + (d/4)*32 + 4*(j%8) + (d%4)] == B[d, j]`, band `j/8`.
/// OOM fallback only.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn unpack_b_i16_sme(data: &[i16], n: usize, k: usize) -> Vec<i16> {
    let per_tile = k.div_ceil(4) * 32;
    let mut b = vec![0i16; k * n];
    for j in 0..n {
        let (band, lane) = (j / 8, j % 8);
        for d in 0..k {
            b[d * n + j] = data[band * per_tile + (d / 4) * 32 + 4 * lane + (d % 4)];
        }
    }
    b
}

/// Unpack an SME-packed i8 B buffer (`gemm_sme_i8i32_packb`'s 4-way-interleaved
/// band layout) back to a row-major `k x n` buffer. Column `j` lives in band
/// `j / 16` (lane `j % 16`); depth `d` is group `d / 4`, slice `d % 4`:
/// `b_pack[band*per_tile + (d/4)*64 + 4*(j%16) + (d%4)] == B[d, j]`. OOM fallback
/// only. `per_tile == ceil(k/4) * 64`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn unpack_b_i8_sme(data: &[i8], n: usize, k: usize) -> Vec<i8> {
    let kp4 = k.div_ceil(4);
    let per_tile = kp4 * 64;
    let mut b = vec![0i8; k * n];
    for j in 0..n {
        let band = j / 16;
        let lane = j % 16;
        for d in 0..k {
            b[d * n + j] = data[band * per_tile + (d / 4) * 64 + 4 * lane + (d % 4)];
        }
    }
    b
}

#[cfg(test)]
mod packb_roundtrip_tests {
    use super::{unpack_b_f32_sme, unpack_b_f64_sme, unpack_b_i8_sme, unpack_b16_sme};
    use crate::{prepack_f16, prepack_f32, prepack_f64, prepack_i8};
    use half::f16;

    // Ragged (n, k) incl. non-multiples of the 32-wide (f16) / 16-wide (i8) tile.
    const SHAPES: &[(usize, usize)] = &[(40, 7), (33, 16), (64, 3), (17, 9)];

    #[test]
    fn unpack_b16_sme_inverts_prepack_f16() {
        for &(n, k) in SHAPES {
            let mut s = 0xb16b_5151_dead_0001u64 ^ ((n * 131 + k * 17) as u64);
            let b: Vec<f16> = (0..k * n)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    f16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                })
                .collect();
            let packed = prepack_f16(&b, n, k);
            // Off-SME `data` is a plain row-major copy: the tile-major unpack
            // inverse does not apply, so only assert when actually SME-packed.
            if !packed.sme {
                continue;
            }
            let got = unpack_b16_sme(&packed.data, n, k);
            assert_eq!(got.len(), b.len(), "len mismatch n={n} k={k}");
            for (idx, (&g, w)) in got.iter().zip(&b).enumerate() {
                assert_eq!(
                    g,
                    w.to_bits(),
                    "unpack_b16_sme n={n} k={k} idx={idx}: round-trip diverges"
                );
            }
        }
    }

    #[test]
    fn unpack_b_band_inverts_prepack_f32_f64() {
        for &(n, k) in SHAPES {
            let b: Vec<f32> = (0..k * n).map(|i| (i % 97) as f32 * 0.25 - 12.0).collect();
            let p = prepack_f32(&b, n, k);
            if p.sme {
                assert_eq!(unpack_b_f32_sme(&p.data, n, k), b, "f32 n={n} k={k}");
            }
            let b64: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
            let p64 = prepack_f64(&b64, n, k);
            if p64.sme {
                assert_eq!(unpack_b_f64_sme(&p64.data, n, k), b64, "f64 n={n} k={k}");
            }
        }
    }

    #[test]
    fn unpack_b_i8_sme_inverts_prepack_i8() {
        for &(n, k) in SHAPES {
            let mut s = 0x18b1_2727_dead_0001u64 ^ ((n * 131 + k * 17) as u64);
            let b: Vec<i8> = (0..k * n)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    ((s >> 48) as i32 % 255 - 127) as i8
                })
                .collect();
            let packed = prepack_i8(&b, n, k);
            if !packed.sme {
                continue;
            }
            let got = unpack_b_i8_sme(&packed.data, n, k);
            assert_eq!(got, b, "unpack_b_i8_sme n={n} k={k}: round-trip diverges");
        }
    }
}

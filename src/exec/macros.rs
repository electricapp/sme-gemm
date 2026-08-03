//! The macros that generate the 16-bit (f16/bf16) batched and packed
//! fused-epilogue paths, which differ only in kernel symbol and fallback.

macro_rules! batched_ep_impl {
    ($name:ident, $T:ty, $cap:ident, $run:ident, $ref:path, $guard:expr) => {
        #[allow(clippy::too_many_arguments, unreachable_pub)]
        pub fn $name(
            a: &[$T],
            b: &[$T],
            c: &mut [$T],
            count: usize,
            m: usize,
            n: usize,
            k: usize,
            ep: &$crate::epilogue::Epilogue<'_, $T>,
        ) {
            ($guard)(ep);
            $crate::exec::validate_ep(ep, m, n);
            if m == 0 || n == 0 || count == 0 {
                return;
            }
            assert_eq!(
                a.len(),
                $crate::exec::checked_dims(count, m, k),
                "a is count*m*k"
            );
            assert_eq!(
                b.len(),
                $crate::exec::checked_dims(count, k, n),
                "b is count*k*n"
            );
            assert_eq!(
                c.len(),
                $crate::exec::checked_dims(count, m, n),
                "c is count*m*n"
            );

            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if $crate::probe::caps().$cap {
                let fnodes = $crate::exec::resolve_nodes(&ep.nodes, ep.act, n);
                let desc = $crate::ffi::EpDesc16 {
                    n_nodes: u32::try_from(fnodes.len())
                        .expect("epilogue op-graph exceeds u32 nodes"),
                    nodes: fnodes.as_ptr(),
                };
                // SAFETY: contiguous row-major batches of `count` items, item
                // strides m*k / k*n / m*n; the kernel reads a/b and writes c in
                // bounds. The shared node array (and its operand slices) outlive
                // the call. Operands were length-validated against m/n.
                let rc = unsafe {
                    $crate::ffi::$run(
                        count,
                        m,
                        n,
                        k,
                        c.as_mut_ptr().cast::<u16>(),
                        a.as_ptr().cast::<u16>(),
                        b.as_ptr().cast::<u16>(),
                        &raw const desc,
                    )
                };
                // rc != 0: an allocation failed and c is untouched; fall through
                // to the per-item reference for a correct result.
                if rc == 0 {
                    return;
                }
            }
            for i in 0..count {
                let ci = &mut c[i * m * n..(i + 1) * m * n];
                $ref(
                    m,
                    n,
                    k,
                    ci,
                    n,
                    1,
                    &a[i * m * k..(i + 1) * m * k],
                    k,
                    1,
                    &b[i * k * n..(i + 1) * k * n],
                    n,
                    1,
                    <$T as $crate::exec::PackedEpilogue>::ep_from_f32(0.0),
                    <$T as $crate::exec::PackedEpilogue>::ep_from_f32(1.0),
                );
                $crate::exec::apply_ep_scalar(ci, m, n, n, 1, ep);
            }
        }
    };
}
pub(crate) use batched_ep_impl;

/// Generates the core packed-epilogue path for a 16-bit dtype (`u16`-backed):
/// `D = act(beta*(A@B) + composable bias)` to a row- or column-major output. f16
/// and bf16 differ only in the kernel symbol, the reference fallback, and (bf16)
/// the activation restriction, so the body is shared here. `$guard` runs the
/// per-dtype activation check.
macro_rules! packed_ep_impl {
    ($name:ident, $T:ty, $run:ident, $ref:path, $guard:expr) => {
        #[allow(unreachable_pub)]
        pub fn $name(
            a: &[$T],
            packed: &$crate::element::Packed<$T>,
            c: &mut [$T],
            m: usize,
            ep: &$crate::epilogue::Epilogue<'_, $T>,
            beta: $T,
            col_major: bool,
        ) {
            let (n, k) = (packed.n, packed.k);
            assert_eq!(a.len(), $crate::exec::checked_dim2(m, k));
            assert_eq!(c.len(), $crate::exec::checked_dim2(m, n));
            ($guard)(ep);
            $crate::exec::validate_ep(ep, m, n);
            if m == 0 || n == 0 {
                return;
            }
            let (dst_cs, dst_rs): (isize, isize) = if col_major {
                (m as isize, 1)
            } else {
                (1, n as isize)
            };
            let has_ep = !ep.nodes.is_empty() || ep.act.is_some();
            // On the OOM fallback this owns an unpacked row-major B; otherwise the
            // scalar fallback borrows `packed.data` directly (sme == false).
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let mut unpacked: Vec<u16> = Vec::new();
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let fnodes = $crate::exec::resolve_nodes(&ep.nodes, ep.act, n);
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if packed.sme {
                let desc = $crate::ffi::EpDesc16 {
                    n_nodes: u32::try_from(fnodes.len())
                        .expect("epilogue op-graph exceeds u32 nodes"),
                    nodes: fnodes.as_ptr(),
                };
                // NULL when no epilogue work, so an unscaled row-major call keeps
                // the single-instruction pure-store fast path.
                let ep_ptr = if has_ep {
                    &raw const desc
                } else {
                    core::ptr::null()
                };
                // SAFETY: data packed for (n,k); a is row-major m*k; c strides
                // describe the chosen layout; desc.bias (if set) has n readable.
                let rc = unsafe {
                    $crate::ffi::$run(
                        m,
                        n,
                        k,
                        c.as_mut_ptr().cast::<u16>(),
                        dst_cs,
                        dst_rs,
                        0,
                        a.as_ptr().cast::<u16>(),
                        1,
                        k as isize,
                        packed.data.as_ptr(),
                        0,
                        beta.to_bits(),
                        ep_ptr,
                    )
                };
                if rc == 0 {
                    return;
                }
                // rc != 0: an allocation failed and c is untouched. Unpack B to
                // row-major and drop into the scalar fallback below.
                unpacked = $crate::exec::unpack_b16_sme(&packed.data, n, k);
            }
            // Fallback: unpacked GEMM (beta scale, overwrite) at the chosen
            // layout, then the scalar epilogue indexed by the same strides.
            // SAFETY: f16/bf16 are repr(transparent) over u16; the backing buffer
            // (the row-major non-SME `packed.data`, or the unpacked OOM buffer) is
            // the row-major k*n bit pattern, so reinterpreting as &[$T] is sound.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let (b_ptr, b_len) = if unpacked.is_empty() {
                (packed.data.as_ptr(), packed.data.len())
            } else {
                (unpacked.as_ptr(), unpacked.len())
            };
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let (b_ptr, b_len) = (packed.data.as_ptr(), packed.data.len());
            // SAFETY: f16/bf16 are repr(transparent) over u16 and the buffer holds
            // exactly b_len u16 bit patterns, so the reinterpret is valid.
            let b: &[$T] = unsafe { core::slice::from_raw_parts(b_ptr.cast::<$T>(), b_len) };
            let (rs, cs) = (dst_rs as usize, dst_cs as usize);
            $ref(
                m,
                n,
                k,
                c,
                rs,
                cs,
                a,
                k,
                1,
                b,
                n,
                1,
                <$T as $crate::exec::PackedEpilogue>::ep_from_f32(0.0),
                beta,
            );
            if has_ep {
                $crate::exec::apply_ep_scalar(c, m, n, rs, cs, ep);
            }
        }
    };
}
pub(crate) use packed_ep_impl;

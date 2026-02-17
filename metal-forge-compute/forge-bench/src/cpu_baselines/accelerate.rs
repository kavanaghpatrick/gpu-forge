//! Accelerate BLAS FFI for CPU baseline comparison.
//!
//! Uses Apple's Accelerate framework cblas_sgemm for matrix multiply.

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transA: i32,
        transB: i32,
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: f32,
        C: *mut f32,
        ldc: i32,
    );
}

/// CblasRowMajor ordering constant.
const CBLAS_ROW_MAJOR: i32 = 101;
/// CblasNoTrans (no transpose) constant.
const CBLAS_NO_TRANS: i32 = 111;

/// Perform C = alpha * A * B + beta * C using Accelerate cblas_sgemm.
///
/// A is MxK, B is KxN, C is MxN. All row-major.
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "A must be M*K elements");
    assert_eq!(b.len(), k * n, "B must be K*N elements");
    assert_eq!(c.len(), m * n, "C must be M*N elements");

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0, // alpha
            a.as_ptr(),
            k as i32, // lda = K for row-major
            b.as_ptr(),
            n as i32, // ldb = N for row-major
            0.0, // beta
            c.as_mut_ptr(),
            n as i32, // ldc = N for row-major
        );
    }
}

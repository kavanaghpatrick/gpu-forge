//! Shared types matching Metal shader structs.
//!
//! All param structs are #[repr(C)] and 16-byte aligned to match
//! the Metal Shading Language struct layout.

/// Parameters for reduce kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReduceParams {
    pub element_count: u32,
    pub _pad: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_params_layout() {
        assert_eq!(
            std::mem::size_of::<ReduceParams>(),
            16,
            "ReduceParams must be 16 bytes"
        );
        assert_eq!(
            std::mem::align_of::<ReduceParams>(),
            4,
            "ReduceParams must be 4-byte aligned"
        );

        // Verify field offsets
        let p = ReduceParams {
            element_count: 0,
            _pad: [0; 3],
        };
        let base = &p as *const _ as usize;
        let element_count_offset =
            &p.element_count as *const _ as usize - base;
        assert_eq!(element_count_offset, 0, "element_count at offset 0");
    }
}

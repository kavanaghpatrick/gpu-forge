//! Single-command-buffer pipeline builder for composing filter + sort + gather.
//!
//! All dispatches are encoded into **one** compute encoder on **one** command buffer.
//! Sequential dispatches within a single encoder have implicit Metal memory barriers,
//! so the GPU sees each stage's writes before the next stage reads (~1us per dispatch
//! overhead vs ~97us per command buffer).
//!
//! # Example
//!
//! ```no_run
//! use forge_runtime::pipeline::Pipeline;
//! use forge_runtime::ForgeContext;
//! use forge_sort::GpuSorter;
//! use forge_filter::{GpuFilter, Predicate};
//! use forge_runtime::GpuGather;
//!
//! let ctx = ForgeContext::new();
//! let mut sorter = GpuSorter::with_context(ctx.device(), ctx.queue()).unwrap();
//! let mut filter = GpuFilter::with_context(ctx.device(), ctx.queue()).unwrap();
//!
//! let mut pipeline = Pipeline::new(&ctx).unwrap();
//! // pipeline.filter(&mut filter, &input_buf, n, &pred).unwrap();
//! // pipeline.sort(&mut sorter, &sort_buf).unwrap();
//! // pipeline.execute().unwrap();
//! ```

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder,
};

use forge_filter::{FilterKey, GpuFilter, Predicate, PendingFilterResult};
use forge_sort::{GpuSorter, SortBuffer, SortError};

use crate::context::ForgeContext;
use crate::gather::GpuGather;

/// Error type for pipeline operations.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("failed to create command buffer")]
    CommandBuffer,
    #[error("failed to create compute encoder")]
    Encoder,
    #[error("GPU execution failed: {0}")]
    GpuExecution(String),
    #[error("sort error: {0}")]
    Sort(#[from] SortError),
    #[error("filter error: {0}")]
    Filter(#[from] forge_filter::FilterError),
}

/// Single-command-buffer pipeline that composes filter, sort, and gather dispatches.
///
/// Created via [`Pipeline::new`], which allocates one command buffer and one compute
/// encoder from the given [`ForgeContext`]. Call [`filter`](Pipeline::filter),
/// [`sort`](Pipeline::sort), and [`gather`](Pipeline::gather) to encode stages,
/// then [`execute`](Pipeline::execute) to commit and wait.
pub struct Pipeline {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Pipeline {
    /// Create a new pipeline with a fresh command buffer and compute encoder.
    pub fn new(ctx: &ForgeContext) -> Result<Self, PipelineError> {
        let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> = ctx.queue();
        let cmd = queue.commandBuffer().ok_or(PipelineError::CommandBuffer)?;
        let encoder = cmd
            .computeCommandEncoder()
            .ok_or(PipelineError::Encoder)?;
        Ok(Self { cmd, encoder })
    }

    /// Encode a filter stage onto this pipeline's encoder.
    ///
    /// Returns a [`PendingFilterResult`] whose count is only valid **after**
    /// [`execute`](Pipeline::execute) completes. Call
    /// [`PendingFilterResult::resolve`] after execute to get the final
    /// [`FilterResult`](forge_filter::FilterResult).
    pub fn filter<T: FilterKey>(
        &self,
        gpu_filter: &mut GpuFilter,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
        pred: &Predicate<T>,
    ) -> Result<PendingFilterResult<T>, PipelineError> {
        let pending = gpu_filter.encode_filter(&self.encoder, input_buf, n, pred)?;
        Ok(pending)
    }

    /// Encode a sort stage onto this pipeline's encoder.
    ///
    /// The `buf` must already contain the data to sort (e.g., from a prior filter
    /// stage converted via [`filter_result_to_sort_buffer`](crate::filter_result_to_sort_buffer)).
    pub fn sort(
        &self,
        gpu_sorter: &mut GpuSorter,
        buf: &SortBuffer<u32>,
    ) -> Result<(), PipelineError> {
        gpu_sorter.encode_sort(&self.encoder, buf)?;
        Ok(())
    }

    /// Encode a gather stage onto this pipeline's encoder.
    ///
    /// Gathers `count` elements from `source` at positions given by `indices`
    /// into `output`. Set `is_64bit` for u64 element types.
    pub fn gather(
        &self,
        gpu_gather: &GpuGather,
        source: &ProtocolObject<dyn MTLBuffer>,
        indices: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        count: u32,
        is_64bit: bool,
    ) {
        gpu_gather.encode_gather(&self.encoder, source, indices, output, count, is_64bit);
    }

    /// End encoding, commit the command buffer, and wait until GPU completes.
    ///
    /// After this returns successfully, any [`PendingFilterResult`] obtained from
    /// [`filter`](Pipeline::filter) can be resolved via
    /// [`PendingFilterResult::resolve`].
    pub fn execute(self) -> Result<(), PipelineError> {
        self.encoder.endEncoding();
        self.cmd.commit();
        self.cmd.waitUntilCompleted();

        if self.cmd.status() == MTLCommandBufferStatus::Error {
            return Err(PipelineError::GpuExecution(
                "command buffer completed with error status".into(),
            ));
        }
        Ok(())
    }
}

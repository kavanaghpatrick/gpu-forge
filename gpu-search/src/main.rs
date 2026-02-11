use gpu_search::gpu::device::GpuDevice;

fn main() {
    println!("gpu-search v0.1.0");
    let gpu = GpuDevice::new();
    gpu.print_info();
}

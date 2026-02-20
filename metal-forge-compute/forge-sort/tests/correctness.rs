use forge_sort::GpuSorter;
use rand::Rng;

fn sort_and_verify(n: usize) {
    let mut rng = rand::thread_rng();
    let data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();
    sort_and_verify_data(data);
}

fn sort_and_verify_data(data: Vec<u32>) {
    let n = data.len();
    let mut expected = data.clone();
    expected.sort();

    let mut actual = data;
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u32(&mut actual).unwrap();

    assert_eq!(
        actual, expected,
        "Sort mismatch at n={}. First diff at index {}",
        n,
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b).unwrap_or(n)
    );
}

// Size tests
#[test] fn test_sort_1k()   { sort_and_verify(1_000); }
#[test] fn test_sort_4k()   { sort_and_verify(4_000); }
#[test] fn test_sort_16k()  { sort_and_verify(16_000); }
#[test] fn test_sort_64k()  { sort_and_verify(64_000); }
#[test] fn test_sort_256k() { sort_and_verify(256_000); }
#[test] fn test_sort_1m()   { sort_and_verify(1_000_000); }
#[test] fn test_sort_4m()   { sort_and_verify(4_000_000); }
#[test] fn test_sort_16m()  { sort_and_verify(16_000_000); }

// Edge cases
#[test]
fn test_empty() {
    let mut data: Vec<u32> = vec![];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u32(&mut data).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_single() {
    let mut data = vec![42u32];
    let mut sorter = GpuSorter::new().unwrap();
    sorter.sort_u32(&mut data).unwrap();
    assert_eq!(data, vec![42]);
}

#[test]
fn test_all_zeros() {
    sort_and_verify_data(vec![0u32; 1_000_000]);
}

#[test]
fn test_all_same() {
    sort_and_verify_data(vec![0xDEADBEEFu32; 1_000_000]);
}

#[test]
fn test_pre_sorted() {
    sort_and_verify_data((0..1_000_000u32).collect());
}

#[test]
fn test_reverse_sorted() {
    sort_and_verify_data((0..1_000_000u32).rev().collect());
}

#[test]
fn test_non_tile_aligned() {
    sort_and_verify(4097); // 4096 + 1
}

#[test]
fn test_sub_tile() {
    sort_and_verify(100);
}

// Buffer reuse test
#[test]
fn test_buffer_reuse() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut rng = rand::thread_rng();

    // Sort 1M
    let mut data1: Vec<u32> = (0..1_000_000).map(|_| rng.gen()).collect();
    let mut expected1 = data1.clone();
    expected1.sort();
    sorter.sort_u32(&mut data1).unwrap();
    assert_eq!(data1, expected1);

    // Sort 1M again (reuse buffers)
    let mut data2: Vec<u32> = (0..1_000_000).map(|_| rng.gen()).collect();
    let mut expected2 = data2.clone();
    expected2.sort();
    sorter.sort_u32(&mut data2).unwrap();
    assert_eq!(data2, expected2);

    // Sort smaller (reuse larger buffers)
    let mut data3: Vec<u32> = (0..100_000).map(|_| rng.gen()).collect();
    let mut expected3 = data3.clone();
    expected3.sort();
    sorter.sort_u32(&mut data3).unwrap();
    assert_eq!(data3, expected3);
}

// Zero-copy SortBuffer tests
#[test]
fn test_sort_buffer_basic() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer(1_000_000);

    let mut rng = rand::thread_rng();
    let data: Vec<u32> = (0..1_000_000).map(|_| rng.gen()).collect();
    let mut expected = data.clone();
    expected.sort();

    buf.copy_from_slice(&data);
    sorter.sort_buffer(&buf).unwrap();

    assert_eq!(buf.as_slice(), &expected[..]);
}

#[test]
fn test_sort_buffer_direct_write() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer(10);

    // Write directly into GPU memory
    let slice = buf.as_mut_slice();
    slice[0] = 9;
    slice[1] = 3;
    slice[2] = 7;
    slice[3] = 1;
    slice[4] = 5;
    buf.set_len(5);

    sorter.sort_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &[1, 3, 5, 7, 9]);
}

#[test]
fn test_sort_buffer_reuse() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer(100_000);
    let mut rng = rand::thread_rng();

    for _ in 0..5 {
        let data: Vec<u32> = (0..100_000).map(|_| rng.gen()).collect();
        let mut expected = data.clone();
        expected.sort();

        buf.copy_from_slice(&data);
        sorter.sort_buffer(&buf).unwrap();
        assert_eq!(buf.as_slice(), &expected[..]);
    }
}

#[test]
fn test_sort_buffer_16m() {
    let mut sorter = GpuSorter::new().unwrap();
    let n = 16_000_000;
    let mut buf = sorter.alloc_sort_buffer(n);

    let mut rng = rand::thread_rng();
    let data: Vec<u32> = (0..n).map(|_| rng.gen()).collect();

    buf.copy_from_slice(&data);

    let start = std::time::Instant::now();
    sorter.sort_buffer(&buf).unwrap();
    let elapsed = start.elapsed();

    // Verify sorted
    let result = buf.as_slice();
    for i in 1..result.len() {
        assert!(result[i - 1] <= result[i], "Not sorted at index {}", i);
    }

    // Zero-copy should be faster than sort_u32 (no memcpy overhead)
    assert!(
        elapsed.as_millis() < 30,
        "sort_buffer 16M took {}ms, expected < 30ms",
        elapsed.as_millis()
    );
}

#[test]
fn test_sort_buffer_empty() {
    let mut sorter = GpuSorter::new().unwrap();
    let buf = sorter.alloc_sort_buffer(100);
    // len=0 by default
    assert!(buf.is_empty());
    sorter.sort_buffer(&buf).unwrap();
}

#[test]
fn test_sort_buffer_single() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut buf = sorter.alloc_sort_buffer(10);
    buf.copy_from_slice(&[42]);
    sorter.sort_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &[42]);
}

#[test]
fn test_sort_buffer_and_sort_u32_interleaved() {
    let mut sorter = GpuSorter::new().unwrap();
    let mut rng = rand::thread_rng();

    // sort_buffer first
    let mut buf = sorter.alloc_sort_buffer(10_000);
    let data1: Vec<u32> = (0..10_000).map(|_| rng.gen()).collect();
    let mut exp1 = data1.clone();
    exp1.sort();
    buf.copy_from_slice(&data1);
    sorter.sort_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &exp1[..]);

    // Then sort_u32
    let mut data2: Vec<u32> = (0..50_000).map(|_| rng.gen()).collect();
    let mut exp2 = data2.clone();
    exp2.sort();
    sorter.sort_u32(&mut data2).unwrap();
    assert_eq!(data2, exp2);

    // Then sort_buffer again
    let data3: Vec<u32> = (0..10_000).map(|_| rng.gen()).collect();
    let mut exp3 = data3.clone();
    exp3.sort();
    buf.copy_from_slice(&data3);
    sorter.sort_buffer(&buf).unwrap();
    assert_eq!(buf.as_slice(), &exp3[..]);
}

// Performance sanity test
#[test]
fn test_sort_16m_perf_sanity() {
    let mut rng = rand::thread_rng();
    let mut data: Vec<u32> = (0..16_000_000).map(|_| rng.gen()).collect();
    let mut sorter = GpuSorter::new().unwrap();

    let start = std::time::Instant::now();
    sorter.sort_u32(&mut data).unwrap();
    let elapsed = start.elapsed();

    // Very generous bound: expect ~3ms GPU, assert < 50ms total (includes memcpy)
    assert!(
        elapsed.as_millis() < 50,
        "16M sort took {}ms, expected < 50ms",
        elapsed.as_millis()
    );

    // Verify result is sorted
    for i in 1..data.len() {
        assert!(data[i - 1] <= data[i], "Not sorted at index {}", i);
    }
}

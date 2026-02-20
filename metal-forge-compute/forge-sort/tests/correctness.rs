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

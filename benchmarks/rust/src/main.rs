//! Measure repeated Rust reference mmap, validation, and first-tensor access.

use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};

const FIRST_TENSOR: &str = "tensor_000";

fn touch_first(path: &Path) -> Result<usize, Box<dyn Error>> {
    let file = File::open(path)?;
    // SAFETY: the benchmark treats its generated archive as immutable for the
    // lifetime of the mapping, matching the reference crate's documented use.
    let mapping = unsafe { MmapOptions::new().map(&file)? };
    let archive = SafeTensors::deserialize(&mapping)?;
    let tensor = archive.tensor(FIRST_TENSOR)?;
    if tensor.dtype() != Dtype::F32 || tensor.shape() != [1] {
        return Err("the first benchmark tensor must have shape [1] and dtype F32".into());
    }
    let bytes: [u8; 4] = tensor
        .data()
        .try_into()
        .map_err(|_| "the first benchmark tensor must contain one F32 value")?;
    Ok(usize::from(f32::from_le_bytes(bytes) != 0.0))
}

fn parse_count(value: &str, name: &str, allow_zero: bool) -> Result<usize, Box<dyn Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a non-negative integer"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(parsed)
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() == 2 {
        println!("{}", touch_first(Path::new(&arguments[1]))?);
        return Ok(());
    }
    if arguments.len() != 4 {
        return Err("usage: map-first-rust <archive> [<warmups> <samples>]".into());
    }

    let path = Path::new(&arguments[1]);
    let warmups = parse_count(&arguments[2], "warmups", true)?;
    let sample_count = parse_count(&arguments[3], "samples", false)?;
    let mut checksum = 0;
    for _ in 0..warmups {
        checksum += touch_first(path)?;
    }

    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let started = Instant::now();
        checksum += touch_first(path)?;
        samples.push(started.elapsed().as_nanos());
    }

    for sample in samples {
        println!("sample_ns {sample}");
    }
    println!("checksum {checksum}");
    Ok(())
}

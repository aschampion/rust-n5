# N5 [![Build Status](https://travis-ci.org/aschampion/rust-n5.svg?branch=master)](https://travis-ci.org/aschampion/rust-n5)

A (mostly pure) rust implementation of the [N5 "Not HDF5" n-dimensional tensor file system storage format](https://github.com/saalfeldlab/n5) created by the Saalfeld lab at Janelia Research Campus.

Compatible with Java N5 Version 2.0.2.

## Differences from Java N5
- Dataset paths are relative. The root path in a dataset is `""`, not `"/"`.
- Dataset paths are more strict. Calling methods with paths outside the dataset, e.g., `".."`, will return a `Result::Err`.

## Quick start

```toml
[dependencies]
n5 = "0.2"
```

```rust
extern crate n5;

use n5::{
    DatasetAttributes,
    DataBlock,
    DataType,
    N5Reader,
    N5Writer,
    VecDataBlock
};
use n5::compression::CompressionType;
use n5::filesystem::N5Filesystem;

fn n5_roundtrip(root_path: &str) -> std::io::Result<()> {
    let n = N5Filesystem::open_or_create(root_path)?;

    let block_size = vec![44i32, 33, 22];
    let data_attrs = DatasetAttributes::new(
        vec![100, 200, 300],
        block_size.clone(),
        DataType::INT16,
        CompressionType::default(),
    );
    let numel = block_size.iter().product::<i32>() as usize;
    let block_data: Vec<i16> = vec![0i16; numel];

    let block_in = VecDataBlock::new(
        block_size,
        vec![0, 0, 0],
        block_data.clone());

    let path_name = "test/dataset/group";

    n.create_dataset(path_name, &data_attrs)?;
    n.write_block(path_name, &data_attrs, &block_in)?;

    let block_out = n.read_block::<i16>(path_name, &data_attrs, vec![0, 0, 0])?
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data);

    Ok(())
}

fn main() {
    n5_roundtrip("tmp.n5").expect("N5 roundtrip failed!");
}
```

## Major TODOs
- Easy import prelude
- Kludge `DefaultBlock` type
- Generally, direct translation from Java is unidiomatic and a mess of boxes

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

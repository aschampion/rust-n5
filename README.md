# N5

A (mostly pure) rust implementation of the [N5 "Not HDF5" n-dimensional tensor file system storage format](https://github.com/saalfeldlab/n5) created by the Saalfeld lab at Janelia Research Campus.

**NOTE: THIS IMPLEMENTATION IS NOT YET FUNCTIONAL. THIS NOTICE WILL BE REMOVED ONCE IT IS.**

## Differences from Java N5
- Dataset paths are relative. The root path in a dataset is `""`, not `"/"`.
- Dataset path are more strict. Calling methods with paths outside the dataset, e.g., `".."`, will return a `Result::Err`.

## Major TODOs
- No writing
- Mixture of `T` parameter as scalar type versus `Vec<T>`
- Kludge `Foo` type
- Generally, direct translation from Java is unidiomatic and a mess of boxes

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

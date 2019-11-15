# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2019-11-15
### Changed
- The minimum supported Rust version is now 1.39.
- All coordinates are now unsigned rather than signed integers, since Java N5
  has adopted this recommendation as of spec version 2.1.3.
- `SliceDataBlock` trait allows using slices for `write_block` and
  `read_block_into`.
- `ReadableDataBlock`, `ReinitDataBlock`, and `WriteableDataBlock` traits and
  bounds have been refactored to allow writing of const slices, reinitialization
  without reserialization, and other features.
- `ReflectedType` now has more bounds for thread safety.
- LZ4 blocks are now written in independent mode, to more closely match the
  behavior of Java N5.
- `read_ndarray` now performs fewer allocations.


## [0.4.0] - 2019-06-05
### Added
- `N5Reader::read_block_into`: read a block into an existing `VecDataBlock`
  without allocating a new block.
- `data_type_match!`: a macro to dispatch a primitive-type generic block of
  code based on a `DataType`.
- `ReflectedType` trait, which supercedes the `TypeReflection` trait,
  `DataBlockCreator` trait, and `Clone` bound on primitive types.
- `N5NdarrayWriter` supertrait that provides a `write_ndarray` method to write
  ndarrays serially to blocks.

### Changed
- All coordinates are now `SmallVec` types `GridCoord` and `BlockCoord`. This
  avoids allocations for datasets with <= 6 dimensions.
- ndarray reading is now in a `N5NdarrayReader` supertrait.

### Removed
- `TypeReflection` trait.
- `DataBlockCreator` trait.


## [0.3.0] - 2019-01-16
### Changed
- `DataType` implements `Display`.
- `VecDataBlock<T>` implements `Clone` and requires `T: Clone`.


## [0.2.4] - 2018-10-17
### Changed
- Updated the `flate2-rs` GZIP dependency to be compatible with WebAssembly.


## [0.2.3] - 2018-10-11
### Added
- `N5Reader::block_metadata`: retrieve block metadata (currently timestamps)
  without reading the block.


## [0.2.2] - 2018-10-07
### Added
- `N5Reader::get_block_uri`: implementation-specific URI strings for data
  blocks.
- `DatasetAttributes::get_[block_]num_elements`: convenient access to
  dataset and block element counts.
- `DatasetAttributes::coord_iter`: convenient iteration over all possible
  coordinates (requires `use_ndarray` feature).

### Changed
- Filesystem implementation is now behind a `filesystem` feature flag, which is
  default.


## [0.2.1] - 2018-06-18
### Added
- Easy import prelude: `use n5::prelude::*;`

### Fixed
- Mode flag was inverted from correct setting for default and varlength blocks.


## [0.2.0] - 2018-03-10
### Added
- Dataset and container removal methods for `N5Writer`.
- `N5Reader::read_ndarray` to read arbitrary bounding box column-major
  `ndarray` arrays from datasets.

### Fixed
- Performance issues with some data types, especially writes.


## [0.1.0] - 2018-02-28

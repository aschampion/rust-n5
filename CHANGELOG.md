# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

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

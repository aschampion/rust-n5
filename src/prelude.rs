//! N5 prelude.
//!
//! This module contains the most used import targets for easy import into
//! client libraries.
//!
//! ```
//! extern crate n5;
//!
//! use n5::prelude::*;
//! # fn main() { }
//! ```

#[doc(no_inline)]
pub use ::{
    BlockCoord,
    BoundingBox,
    DatasetAttributes,
    DataBlock,
    DataBlockMetadata,
    DataType,
    GridCoord,
    N5Reader,
    N5Writer,
    TypeReflection,
    VecDataBlock,
};
#[doc(no_inline)]
pub use ::compression::{
    self,
    CompressionType,
};
#[cfg(feature = "filesystem")]
#[doc(no_inline)]
pub use ::filesystem::N5Filesystem;

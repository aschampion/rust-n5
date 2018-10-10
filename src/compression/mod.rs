//! Compression for block voxel data.

use std;
use std::io::{Read, Write};


pub mod raw;
#[cfg(feature = "brotli")]
pub mod brotli;
#[cfg(feature = "bzip")]
pub mod bzip;
#[cfg(feature = "gzip")]
pub mod gzip;
#[cfg(feature = "lz")]
pub mod lz;
#[cfg(feature = "xz")]
pub mod xz;


/// Common interface for compressing writers and decompressing readers.
pub trait Compression : Default {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<Read + 'a>;

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<Write + 'a>;
}

/// Enumeration of known compression schemes.
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "type")]
pub enum CompressionType {
    Raw(raw::RawCompression),
    #[cfg(feature = "brotli")]
    Brotli2(brotli::Brotli2Compression),
    #[cfg(feature = "bzip")]
    Bzip2(bzip::Bzip2Compression),
    #[cfg(feature = "gzip")]
    Gzip(gzip::GzipCompression),
    #[cfg(feature = "lz")]
    Lz4(lz::Lz4Compression),
    #[cfg(feature = "xz")]
    Xz(xz::XzCompression),
}

impl CompressionType {
    pub fn new<T: Compression>() -> CompressionType
            where CompressionType: std::convert::From<T> {
        T::default().into()
    }
}

impl Default for CompressionType {
    fn default() -> CompressionType {
        CompressionType::new::<raw::RawCompression>()
    }
}

impl Compression for CompressionType {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<Read + 'a> {
        match *self {
            CompressionType::Raw(ref c) => c.decoder(r),

            #[cfg(feature = "brotli")]
            CompressionType::Brotli2(ref c) => c.decoder(r),

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(ref c) => c.decoder(r),

            #[cfg(feature = "gzip")]
            CompressionType::Gzip(ref c) => c.decoder(r),

            #[cfg(feature = "xz")]
            CompressionType::Xz(ref c) => c.decoder(r),

            #[cfg(feature = "lz")]
            CompressionType::Lz4(ref c) => c.decoder(r),
        }
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<Write + 'a> {
        match *self {
            CompressionType::Raw(ref c) => c.encoder(w),

            #[cfg(feature = "brotli")]
            CompressionType::Brotli2(ref c) => c.encoder(w),

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(ref c) => c.encoder(w),

            #[cfg(feature = "gzip")]
            CompressionType::Gzip(ref c) => c.encoder(w),

            #[cfg(feature = "xz")]
            CompressionType::Xz(ref c) => c.encoder(w),

            #[cfg(feature = "lz")]
            CompressionType::Lz4(ref c) => c.encoder(w),
        }
    }
}

impl std::fmt::Display for CompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", match *self {
            CompressionType::Raw(_) => "Raw",

            #[cfg(feature = "brotli")]
            CompressionType::Brotli2(_) => "Brotli2",

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(_) => "Bzip2",

            #[cfg(feature = "gzip")]
            CompressionType::Gzip(_) => "Gzip",

            #[cfg(feature = "xz")]
            CompressionType::Xz(_) => "Xz",

            #[cfg(feature = "lz")]
            CompressionType::Lz4(_) => "Lz4",
        })
    }
}

macro_rules! compression_from_impl {
    ($variant:ident, $c_type:ty) => {
        impl std::convert::From<$c_type> for CompressionType {
            fn from(c: $c_type) -> Self {
                CompressionType::$variant(c)
            }
        }
    }
}

compression_from_impl!(Raw, raw::RawCompression);
#[cfg(feature = "brotli")]
compression_from_impl!(Brotli2, brotli::Brotli2Compression);
#[cfg(feature = "bzip")]
compression_from_impl!(Bzip2, bzip::Bzip2Compression);
#[cfg(feature = "gzip")]
compression_from_impl!(Gzip, gzip::GzipCompression);
#[cfg(feature = "xz")]
compression_from_impl!(Xz, xz::XzCompression);
#[cfg(feature = "lz")]
compression_from_impl!(Lz4, lz::Lz4Compression);

use std::io::{Read, Write};


pub mod raw;
#[cfg(feature = "bzip")]
pub mod bzip;
#[cfg(feature = "gzip")]
pub mod gzip;
#[cfg(feature = "lz")]
pub mod lz;
#[cfg(feature = "xz")]
pub mod xz;


pub trait Compression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<Read + 'a>;

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<Write + 'a>;
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "type")]
pub enum CompressionType {
    Raw(raw::RawCompression),
    #[cfg(feature = "bzip")]
    Bzip2(bzip::Bzip2Compression),
    #[cfg(feature = "gzip")]
    Gzip(gzip::GzipCompression),
    #[cfg(feature = "lz")]
    Lz4(lz::Lz4Compression),
    #[cfg(feature = "xz")]
    Xz(xz::XzCompression),
}

// impl CompressionType {
//     pub fn get_reader<'a, R: Read + 'a>(&self) -> Box<Compression<'a, R>> {
//         #[allow(unreachable_patterns)] // Ignore the default case.
//         match *self {
//             CompressionType::Raw => Box::new(raw::RawCompression),

//             #[cfg(feature = "bzip")]
//             CompressionType::Bzip2(ref params) =>
//                 Box::new(bzip::Bzip2Compression::new(params)),

//             #[cfg(feature = "gzip")]
//             CompressionType::Gzip(ref params) =>
//                 Box::new(gzip::GzipCompression::new(params)),

//             #[cfg(feature = "xz")]
//             CompressionType::Xz(ref params) =>
//                 Box::new(xz::XzCompression::new(params)),

//             #[cfg(feature = "lz")]
//             CompressionType::Lz4(ref params) =>
//                 Box::new(lz::Lz4Compression::new(params)),

//             // Default case to panic if the requested compression feature is not
//             // enabled.
//             _ => unimplemented!(),
//         }
//     }
// }

impl Compression for CompressionType {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<Read + 'a> {
        #[allow(unreachable_patterns)] // Ignore the default case.
        match *self {
            CompressionType::Raw(ref c) => c.decoder(r),

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(ref c) => c.decoder(r),

            #[cfg(feature = "gzip")]
            CompressionType::Gzip(ref c) => c.decoder(r),

            #[cfg(feature = "xz")]
            CompressionType::Xz(ref c) => c.decoder(r),

            #[cfg(feature = "lz")]
            CompressionType::Lz4(ref c) => c.decoder(r),

            // Default case to panic if the requested compression feature is not
            // enabled.
            _ => unimplemented!(),
        }
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<Write + 'a> {
        #[allow(unreachable_patterns)] // Ignore the default case.
        match *self {
            CompressionType::Raw(ref c) => c.encoder(w),

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(ref c) => c.encoder(w),

            #[cfg(feature = "gzip")]
            CompressionType::Gzip(ref c) => c.encoder(w),

            #[cfg(feature = "xz")]
            CompressionType::Xz(ref c) => c.encoder(w),

            #[cfg(feature = "lz")]
            CompressionType::Lz4(ref c) => c.encoder(w),

            // Default case to panic if the requested compression feature is not
            // enabled.
            _ => unimplemented!(),
        }
    }
}

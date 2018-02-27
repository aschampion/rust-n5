use std::io::Read;


pub mod raw;
#[cfg(feature = "bzip")]
pub mod bzip;
#[cfg(feature = "gzip")]
pub mod gzip;
#[cfg(feature = "lz")]
pub mod lz;
#[cfg(feature = "xz")]
pub mod xz;


pub trait Compression<'a, R: Read + 'a> {
    fn decoder(&self, r: R) -> Box<Read + 'a>;
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "type")]
pub enum CompressionType {
    Raw,
    Bzip2(Bzip2Parameters),
    Gzip(GzipParameters),
    Lz4(Lz4Parameters),
    Xz(XzParameters),
}

impl CompressionType {
    pub fn get_reader<'a, R: Read + 'a>(&self) -> Box<Compression<'a, R>> {
        #[allow(unreachable_patterns)] // Ignore the default case.
        match *self {
            CompressionType::Raw => Box::new(raw::RawCompression),

            #[cfg(feature = "bzip")]
            CompressionType::Bzip2(ref params) =>
                Box::new(bzip::Bzip2Compression::new(params)),

            #[cfg(feature = "gzip")]
            CompressionType::Gzip(ref params) =>
                Box::new(gzip::GzipCompression::new(params)),

            #[cfg(feature = "xz")]
            CompressionType::Xz(ref params) =>
                Box::new(xz::XzCompression::new(params)),

            #[cfg(feature = "lz")]
            CompressionType::Lz4(ref params) =>
                Box::new(lz::Lz4Compression::new(params)),

            // Default case to panic if the requested compression feature is not
            // enabled.
            _ => unimplemented!(),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Bzip2Parameters {
    #[serde(default = "default_bzip_block_size")]
    block_size: u8,
}

// Will never understand why serde decided against $expr defaults. Ugh.
fn default_bzip_block_size() -> u8 {8}

impl Default for Bzip2Parameters {
    fn default() -> Bzip2Parameters {
        Bzip2Parameters {
            block_size: default_bzip_block_size(),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GzipParameters {
    #[serde(default = "default_gzip_level")]
    level: i32,
}

fn default_gzip_level() -> i32 {-1}

impl Default for GzipParameters {
    fn default() -> GzipParameters {
        GzipParameters {
            level: default_gzip_level(),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Lz4Parameters {
    #[serde(default = "default_lz4_block_size")]
    block_size: i32,
}

fn default_lz4_block_size() -> i32 {65_536}

impl Default for Lz4Parameters {
    fn default() -> Lz4Parameters {
        Lz4Parameters {
            block_size: default_lz4_block_size(),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct XzParameters {
    #[serde(default = "default_xz_preset")]
    preset: i32,
}

fn default_xz_preset() -> i32 {6}

impl Default for XzParameters {
    fn default() -> XzParameters {
        XzParameters {
            preset: default_xz_preset(),
        }
    }
}

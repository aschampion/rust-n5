use std::io::{Read, Write};

use flate2::Compression as GzCompression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{
    Deserialize,
    Serialize,
};

use super::{
    Compression,
};


#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GzipCompression {
    #[serde(default = "default_gzip_level")]
    level: i32,
}

impl GzipCompression {
    /// Java has -1 as the default compression level for Gzip
    /// despite this not being a valid compression level.
    ///
    /// Use `flate2`'s default level if the configured level is not in [0, 9].
    /// (At the time of writing this is 6.)
    fn get_effective_level(&self) -> GzCompression {
        if self.level < 0 || self.level > 9 {
            GzCompression::default()
        } else {
            GzCompression::new(self.level as u32)
        }
    }
}

fn default_gzip_level() -> i32 {-1}

impl Default for GzipCompression {
    fn default() -> GzipCompression {
        GzipCompression {
            level: default_gzip_level(),
        }
    }
}

impl Compression for GzipCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(GzDecoder::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(GzEncoder::new(w, self.get_effective_level()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    #[test]
    fn test_rw() {
        crate::tests::test_block_compression_rw(CompressionType::Gzip(GzipCompression::default()));
    }
}

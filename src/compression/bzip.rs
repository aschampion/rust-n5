use std::io::{Read, Write};

use bzip2::Compression as BzCompression;
use bzip2::read::BzDecoder;
use bzip2::write::BzEncoder;
use serde::{
    Deserialize,
    Serialize,
};

use super::{
    Compression,
};


#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Bzip2Compression {
    #[serde(default = "default_bzip_block_size")]
    block_size: u8,
}

impl Bzip2Compression {
    /// `bzip2` has fixed enum levels for compression.
    fn get_effective_compression(&self) -> BzCompression {
        if i32::from(self.block_size) <= BzCompression::Fastest as i32 {
            BzCompression::Fastest
        } else if i32::from(self.block_size) <= BzCompression::Default as i32 {
            BzCompression::Default
        } else {
            BzCompression::Best
        }
    }
}

fn default_bzip_block_size() -> u8 {8}

impl Default for Bzip2Compression {
    fn default() -> Bzip2Compression {
        Bzip2Compression {
            block_size: default_bzip_block_size(),
        }
    }
}

impl Compression for Bzip2Compression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(BzDecoder::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(BzEncoder::new(w, self.get_effective_compression()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    #[test]
    fn test_rw() {
        crate::tests::test_block_compression_rw(CompressionType::Bzip2(Bzip2Compression::default()));
    }
}

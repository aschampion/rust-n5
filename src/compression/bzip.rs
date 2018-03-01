use std::io::{Read, Write};

use bzip2::Compression as BzCompression;
use bzip2::read::BzDecoder;
use bzip2::write::BzEncoder;

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
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<Read + 'a> {
        Box::new(BzDecoder::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<Write + 'a> {
        Box::new(BzEncoder::new(w, self.get_effective_compression()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compression::CompressionType;

    // Example from the n5 documentation spec.
    const TEST_BLOCK_I16_BZIP2: [u8; 59] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x42, 0x5a, 0x68, 0x39,
        0x31, 0x41, 0x59, 0x26,
        0x53, 0x59, 0x02, 0x3e,
        0x0d, 0xd2, 0x00, 0x00,
        0x00, 0x40, 0x00, 0x7f,
        0x00, 0x20, 0x00, 0x31,
        0x0c, 0x01, 0x0d, 0x31,
        0xa8, 0x73, 0x94, 0x33,
        0x7c, 0x5d, 0xc9, 0x14,
        0xe1, 0x42, 0x40, 0x08,
        0xf8, 0x37, 0x48,
    ];

    #[test]
    fn test_read_doc_spec_block() {
        ::tests::test_read_doc_spec_block(
            &TEST_BLOCK_I16_BZIP2[..],
            CompressionType::Bzip2(Bzip2Compression::default()));
    }

    #[test]
    fn test_rw() {
        ::tests::test_block_compression_rw(CompressionType::Bzip2(Bzip2Compression::default()));
    }
}

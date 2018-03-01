use std::io::{Read, Write};

use super::Compression;


#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
pub struct RawCompression;

impl Compression for RawCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<Read + 'a> {
        Box::new(r)
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<Write + 'a> {
        Box::new(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compression::CompressionType;

    // Example from the n5 documentation spec.
    const TEST_BLOCK_I16_RAW: [u8; 28] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x00, 0x01,
        0x00, 0x02,
        0x00, 0x03,
        0x00, 0x04,
        0x00, 0x05,
        0x00, 0x06,
    ];

    #[test]
    fn test_read_doc_spec_block() {
        ::tests::test_read_doc_spec_block(
            &TEST_BLOCK_I16_RAW[..],
            CompressionType::Raw(RawCompression));
    }

    #[test]
    fn test_rw() {
        ::tests::test_block_compression_rw(CompressionType::Raw(RawCompression));
    }
}

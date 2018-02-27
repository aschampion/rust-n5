use std::io::Read;

use super::Compression;


pub struct RawCompression;

impl<'a, R: Read + 'a> Compression<'a, R> for RawCompression {
    fn decoder(&self, r: R) -> Box<Read + 'a> {
        Box::new(r)
    }
}

#[cfg(test)]
mod tests {
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
        ::tests::test_read_doc_spec_block(&TEST_BLOCK_I16_RAW[..], CompressionType::Raw);
    }
}

use std::io::Read;

use flate2::read::GzDecoder;

use super::{
    Compression,
    GzipParameters,
};


pub struct GzipCompression;

impl GzipCompression {
    pub fn new(_params: &GzipParameters) -> GzipCompression {
        GzipCompression
    }
}

impl<'a, R: Read + 'a> Compression<'a, R> for GzipCompression {
    fn decoder(&self, r: R) -> Box<Read + 'a> {
        Box::new(GzDecoder::new(r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compression::CompressionType;

    // Example from the n5 documentation spec.
    const TEST_BLOCK_I16_GZIP: [u8; 48] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x1f, 0x8b, 0x08, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x63, 0x60,
        0x64, 0x60, 0x62, 0x60,
        0x66, 0x60, 0x61, 0x60,
        0x65, 0x60, 0x03, 0x00,
        0xaa, 0xea, 0x6d, 0xbf,
        0x0c, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn test_read_doc_spec_block() {
        ::tests::test_read_doc_spec_block(
            &TEST_BLOCK_I16_GZIP[..],
            CompressionType::Gzip(GzipParameters::default()));
    }
}

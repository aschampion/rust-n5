use std::io::Read;

use bzip2::read::BzDecoder;

use super::{
    Compression,
    Bzip2Parameters,
};


pub struct Bzip2Compression;

impl Bzip2Compression {
    pub fn new(_params: &Bzip2Parameters) -> Bzip2Compression {
        Bzip2Compression
    }
}

impl<'a, R: Read + 'a> Compression<'a, R> for Bzip2Compression {
    fn decoder(&self, r: R) -> Box<Read + 'a> {
        Box::new(BzDecoder::new(r))
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
            CompressionType::Bzip2(Bzip2Parameters::default()));
    }
}

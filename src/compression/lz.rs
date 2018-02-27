use std::io::Read;

use lz4::Decoder;

use super::{
    Compression,
    Lz4Parameters,
};


pub struct Lz4Compression;

impl Lz4Compression {
    pub fn new(_params: &Lz4Parameters) -> Lz4Compression {
        Lz4Compression
    }
}

impl<'a, R: Read + 'a> Compression<'a, R> for Lz4Compression {
    fn decoder(&self, r: R) -> Box<Read + 'a> {
        Box::new(Decoder::new(r).expect("TODO: LZ4 returns a result here"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compression::CompressionType;

    const TEST_BLOCK_I16_LZ4: [u8; 47] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x04, 0x22, 0x4d, 0x18,
        0x64, 0x40, 0xa7, 0x0c,
        0x00, 0x00, 0x80, 0x00,
        0x01, 0x00, 0x02, 0x00,
        0x03, 0x00, 0x04, 0x00,
        0x05, 0x00, 0x06, 0x00,
        0x00, 0x00, 0x00, 0x41,
        0x37, 0x33, 0x08,
    ];

    #[test]
    fn test_read_doc_spec_block() {
        ::tests::test_read_doc_spec_block(
            &TEST_BLOCK_I16_LZ4[..],
            CompressionType::Lz4(Lz4Parameters::default()));
    }
}

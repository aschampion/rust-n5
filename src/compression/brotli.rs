use std::io::{Read, Write};

use brotli2::CompressParams;
use brotli2::read::BrotliDecoder;
use brotli2::write::BrotliEncoder;

use super::{
    Compression,
};


#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Brotli2Compression {
    #[serde(default = "default_brotli_quality")]
    quality: u32,
    #[serde(default = "default_brotli_lg_window_size")]
    lg_window_size: u32,
}

impl Brotli2Compression {
    fn get_params(&self) -> CompressParams {
        let mut params = CompressParams::new();
        params
            .quality(self.quality)
            .lgwin(self.lg_window_size);
        params
    }
}

fn default_brotli_quality() -> u32 {11}

fn default_brotli_lg_window_size() -> u32 {22}

impl Default for Brotli2Compression {
    fn default() -> Brotli2Compression {
        Brotli2Compression {
            quality: default_brotli_quality(),
            lg_window_size: default_brotli_lg_window_size(),
        }
    }
}

impl Compression for Brotli2Compression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<Read + 'a> {
        Box::new(BrotliDecoder::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<Write + 'a> {
        Box::new(BrotliEncoder::from_params(w, &self.get_params()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compression::CompressionType;

    // Example from the n5 documentation spec.
    const TEST_BLOCK_I16_BROTLI2: [u8; 32] = [
        0x00, 0x00,
        0x00, 0x03,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x03,
        0x1b, 0x0b, 0x00, 0x00,
        0xa6, 0x01, 0x02, 0x04,
        0x06, 0x86, 0x2d, 0x91,
        0x92, 0x79, 0x02, 0x04,
    ];

    #[test]
    fn test_read_doc_spec_block() {
        ::tests::test_read_doc_spec_block(
            TEST_BLOCK_I16_BROTLI2.as_ref(),
            CompressionType::Brotli2(Brotli2Compression::default()));
    }

    #[test]
    fn test_rw() {
        ::tests::test_block_compression_rw(CompressionType::Brotli2(Brotli2Compression::default()));
    }
}

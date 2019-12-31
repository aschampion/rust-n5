use std::io::{Read, Write};

use serde::{
    Deserialize,
    Serialize,
};
use xz2::read::XzDecoder;
use xz2::write::XzEncoder;

use super::{
    Compression,
};


#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
pub struct XzCompression {
    #[serde(default = "default_xz_preset")]
    preset: i32,
}

fn default_xz_preset() -> i32 {6}

impl Default for XzCompression {
    fn default() -> XzCompression {
        XzCompression {
            preset: default_xz_preset(),
        }
    }
}

impl Compression for XzCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(XzDecoder::new(r))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        // TODO: check that preset is non-negative.s
        Box::new(XzEncoder::new(w, self.preset as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    #[test]
    fn test_rw() {
        crate::tests::test_block_compression_rw(CompressionType::Xz(XzCompression::default()));
    }
}

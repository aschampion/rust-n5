use std::io::{Read, Write};

use serde::{
    Deserialize,
    Serialize,
};

use super::Compression;


#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
pub struct RawCompression;

impl Compression for RawCompression {
    fn decoder<'a, R: Read + 'a>(&self, r: R) -> Box<dyn Read + 'a> {
        Box::new(r)
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    #[test]
    fn test_rw() {
        crate::tests::test_block_compression_rw(CompressionType::Raw(RawCompression));
    }

}

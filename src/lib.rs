extern crate byteorder;
#[cfg(feature = "bzip")]
extern crate bzip2;
#[cfg(feature = "gzip")]
extern crate flate2;
extern crate fs2;
#[macro_use]
extern crate lazy_static;
#[cfg(feature = "lz")]
extern crate lz4;
extern crate serde;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
#[cfg(test)]
extern crate tempdir;
extern crate regex;
#[cfg(feature = "xz")]
extern crate xz2;


use std::io::{
    Error,
    ErrorKind,
};

use byteorder::{
    BigEndian,
    ByteOrder,
    ReadBytesExt,
    WriteBytesExt,
};
use serde::Serialize;

use ::compression::Compression;

pub mod compression;
pub mod filesystem;


lazy_static! {
    static ref VERSION: Version = {
        Version::new(2, 0, 2, "")
    };
}

const VERSION_ATTRIBUTE_KEY: &str = "n5";


pub trait N5Reader {
    fn get_version(&self) -> Result<Version, Error>;

    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error>;

    /// Test whether a group or dataset exists.
    fn exists(&self, path_name: &str) -> bool;

    /// Test whether a dataset exists.
    fn dataset_exists(&self, path_name: &str) -> bool {
        self.exists(path_name) && self.get_dataset_attributes(path_name).is_ok()
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: Vec<i64>,
    ) -> Result<Option<Box<DataBlock<Vec<T>>>>, Error>
        where DataType: DataBlockCreator<Vec<T>>;

    /// List all groups (including datasets) in a group.
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error>;
}

pub trait N5Writer : N5Reader {
    /// Set a single attribute.
    fn set_attribute<T: Serialize>(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        key: String,
        attribute: T,
    ) -> Result<(), Error> {
        self.set_attributes(
            path_name,
            vec![(key, serde_json::to_value(attribute)?)].into_iter().collect())
    }

    /// Set a map of attributes.
    fn set_attributes(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), Error>;

    /// Set mandatory dataset attributes.
    fn set_dataset_attributes(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> Result<(), Error> {
        if let serde_json::Value::Object(map) = serde_json::to_value(data_attrs)? {
            self.set_attributes(path_name, map)
        } else {
            panic!("Impossible: DatasetAttributes serializes to object")
        }
    }

    /// Create a group (directory).
    fn create_group(&self, path_name: &str) -> Result<(), Error>;

    fn create_dataset(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> Result<(), Error> {
        self.create_group(path_name)?;
        self.set_dataset_attributes(path_name, data_attrs)
    }

    fn write_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: Box<DataBlock<T>>,
    ) -> Result<(), Error>;
}


#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
}

pub trait TypeReflection<T> {
    fn get_type_variant() -> Self;
}

// TODO: replace this with a generic inherent function and instead check that
// dataset DataType is expected type (via `TypeReflection` trait).
pub trait DataBlockCreator<T> {
    fn create_data_block(
        &self,
        block_size: Vec<i32>,
        grid_position: Vec<i64>,
        num_el: usize,
    ) -> Option<Box<DataBlock<T>>>;
}

macro_rules! data_type_block_creator {
    ($d_name:ident, $d_type:ty) => {
        impl TypeReflection<$d_type> for DataType {
            fn get_type_variant() -> DataType {
                DataType::$d_name
            }
        }

        impl DataBlockCreator<Vec<$d_type>> for DataType {
            fn create_data_block(
                &self,
                block_size: Vec<i32>,
                grid_position: Vec<i64>,
                num_el: usize,
            ) -> Option<Box<DataBlock<Vec<$d_type>>>> {
                match *self {
                    DataType::$d_name => Some(Box::new(VecDataBlock::<$d_type>::new(
                        block_size,
                        grid_position,
                        vec![0. as $d_type; num_el],
                    ))),
                    _ => None,
                }
            }
        }
    }
}

data_type_block_creator!(UINT8,  u8);
data_type_block_creator!(UINT16, u16);
data_type_block_creator!(UINT32, u32);
data_type_block_creator!(UINT64, u64);
data_type_block_creator!(INT8,  i8);
data_type_block_creator!(INT16, i16);
data_type_block_creator!(INT32, i32);
data_type_block_creator!(INT64, i64);
data_type_block_creator!(FLOAT32, f32);
data_type_block_creator!(FLOAT64, f64);

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DatasetAttributes {
    dimensions: Vec<i64>,
    block_size: Vec<i32>,
    data_type: DataType,
    compression: compression::CompressionType,
}

impl DatasetAttributes {
    pub fn new(
        dimensions: Vec<i64>,
        block_size: Vec<i32>,
        data_type: DataType,
        compression: compression::CompressionType,
    ) -> DatasetAttributes {
        DatasetAttributes {
            dimensions,
            block_size,
            data_type,
            compression,
        }
    }

    pub fn get_dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    pub fn get_block_size(&self) -> &[i32] {
        &self.block_size
    }

    pub fn get_data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn get_compression(&self) -> &compression::CompressionType {
        &self.compression
    }

    pub fn get_ndim(&self) -> usize {
        self.dimensions.len()
    }
}


pub trait ReadableDataBlock {
    /// Unlike Java N5, read the stream directly into the block data instead
    /// of creating a copied byte buffer.
    fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()>;
}

pub trait WriteableDataBlock {
    fn write_data(&self, target: &mut std::io::Write) -> std::io::Result<()>;
}

pub trait DataBlock<T> : ReadableDataBlock + WriteableDataBlock {
    fn get_size(&self) -> &Vec<i32>;

    fn get_grid_position(&self) -> &Vec<i64>;

    fn get_data(&self) -> &T;

    fn get_num_elements(&self) -> i32; // TODO: signed sizes feel awful.
}

pub struct VecDataBlock<T> {
    size: Vec<i32>,
    grid_position: Vec<i64>,
    data: Vec<T>,
}

impl<T> VecDataBlock<T> {
    pub fn new(size: Vec<i32>, grid_position: Vec<i64>, data: Vec<T>) -> VecDataBlock<T> {
        VecDataBlock {
            size,
            grid_position,
            data,
        }
    }
}

macro_rules! vec_data_block_impl {
    ($ty_name:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl ReadableDataBlock for VecDataBlock<$ty_name> {
            fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()> {
                source.$bo_read_fn::<BigEndian>(&mut self.data)
            }
        }

        impl WriteableDataBlock for VecDataBlock<$ty_name> {
            fn write_data(&self, target: &mut std::io::Write) -> std::io::Result<()> {
                const CHUNK: usize = 256;
                let mut buf: [u8; CHUNK * std::mem::size_of::<$ty_name>()] =
                    [0; CHUNK * std::mem::size_of::<$ty_name>()];

                for c in self.data.chunks(CHUNK) {
                    let byte_len = c.len() * std::mem::size_of::<$ty_name>();
                    BigEndian::$bo_write_fn(c, &mut buf[..byte_len]);
                    target.write_all(&buf[..byte_len])?;
                }

                Ok(())
            }
        }
    }
}

vec_data_block_impl!(u16, read_u16_into, write_u16_into);
vec_data_block_impl!(u32, read_u32_into, write_u32_into);
vec_data_block_impl!(u64, read_u64_into, write_u64_into);
vec_data_block_impl!(i16, read_i16_into, write_i16_into);
vec_data_block_impl!(i32, read_i32_into, write_i32_into);
vec_data_block_impl!(i64, read_i64_into, write_i64_into);
vec_data_block_impl!(f32, read_f32_into, write_f32_into);
vec_data_block_impl!(f64, read_f64_into, write_f64_into);

impl ReadableDataBlock for VecDataBlock<u8> {
    fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()> {
        source.read_exact(&mut self.data)
    }
}

impl WriteableDataBlock for VecDataBlock<u8> {
    fn write_data(&self, target: &mut std::io::Write) -> std::io::Result<()> {
        target.write_all(&self.data)
    }
}

impl ReadableDataBlock for VecDataBlock<i8> {
    fn read_data(&mut self, source: &mut std::io::Read) -> std::io::Result<()> {
        // Unsafe necessary here because we need a &mut [u8] to avoid doing
        // individual reads to the i8 data. This is safe.
        let data_ref = unsafe { &mut *(&mut self.data[..] as *mut [i8] as *mut [u8]) };
        source.read_exact(data_ref)
    }
}

impl WriteableDataBlock for VecDataBlock<i8> {
    fn write_data(&self, target: &mut std::io::Write) -> std::io::Result<()> {
        // Unsafe necessary here because we need a &mut [u8] to avoid doing
        // individual writes from the i8 data. This is safe.
        let data_ref = unsafe { &*(&self.data[..] as *const [i8] as *const [u8]) };
        target.write_all(data_ref)
    }
}

impl<T> DataBlock<Vec<T>> for VecDataBlock<T>
        where VecDataBlock<T>: ReadableDataBlock + WriteableDataBlock {
    fn get_size(&self) -> &Vec<i32> {
        &self.size
    }

    fn get_grid_position(&self) -> &Vec<i64> {
        &self.grid_position
    }

    fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    fn get_num_elements(&self) -> i32 {
        self.data.len() as i32
    }
}


pub trait DefaultBlockReader<T, R: std::io::Read> //:
        // BlockReader<Vec<T>, VecDataBlock<T>, R>
        where DataType: DataBlockCreator<Vec<T>> {
    fn read_block(
        mut buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: Vec<i64>,
    ) -> std::io::Result<Box<DataBlock<Vec<T>>>> {
        let mode = buffer.read_i16::<BigEndian>()?;
        let ndim = buffer.read_i16::<BigEndian>()?;
        let mut dims = vec![0; ndim as usize];
        buffer.read_i32_into::<BigEndian>(&mut dims)?;
        let num_el = match mode {
            0 => dims.iter().product(),
            1 => buffer.read_i32::<BigEndian>()?,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Unsupported block mode"))
        };

        let mut block: Box<DataBlock<Vec<T>>> = data_attrs.data_type.create_data_block(
            dims,
            grid_position,
            num_el as usize).unwrap();
        let mut decompressed = data_attrs.compression.decoder(buffer);
        block.read_data(&mut decompressed)?;

        Ok(block)
    }
}

pub trait DefaultBlockWriter<T, W: std::io::Write> {
    fn write_block(
        mut buffer: W,
        data_attrs: &DatasetAttributes,
        block: Box<DataBlock<T>>,
    ) -> std::io::Result<()> {
        let mode: i16 = if block.get_num_elements() == block.get_size().iter().product::<i32>()
            {1} else {0};
        buffer.write_i16::<BigEndian>(mode)?;
        buffer.write_i16::<BigEndian>(data_attrs.get_ndim() as i16)?;
        for i in block.get_size() {
            buffer.write_i32::<BigEndian>(*i)?;
        }

        if mode != 0 {
            buffer.write_i32::<BigEndian>(block.get_num_elements())?;
        }

        let mut compressor = data_attrs.compression.encoder(buffer);
        block.write_data(&mut compressor)?;

        Ok(())
    }
}

// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultBlockReader`, etc.
struct Foo;
impl<T, R: std::io::Read> DefaultBlockReader<T, R> for Foo
        where DataType: DataBlockCreator<Vec<T>> {}
impl<T, W: std::io::Write> DefaultBlockWriter<T, W> for Foo {}


/// A semantic version.
///
/// # Examples
///
/// ```
/// # use n5::Version;
/// # use std::str::FromStr;
/// let v = Version::from_str("1.2.3-suffix").unwrap();
///
/// assert_eq!(v.get_major(), 1);
/// assert_eq!(v.get_minor(), 2);
/// assert_eq!(v.get_patch(), 3);
/// assert_eq!(v.get_suffix(), "-suffix");
/// assert_eq!(v.to_string(), "1.2.3-suffix");
///
/// assert!(v.is_compatible(&Version::from_str("1.1").unwrap()));
/// assert!(!v.is_compatible(&Version::from_str("2.1").unwrap()));
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct Version {
    major: i32,
    minor: i32,
    patch: i32,
    suffix: String,
}

impl Version {
    pub fn new(major: i32, minor: i32, patch: i32, suffix: &str) -> Version {
        Version {
            major,
            minor,
            patch,
            suffix: suffix.to_owned(),
        }
    }

    pub fn get_major(&self) -> i32 {
        self.major
    }

    pub fn get_minor(&self) -> i32 {
        self.minor
    }

    pub fn get_patch(&self) -> i32 {
        self.patch
    }

    pub fn get_suffix(&self) -> &str {
        &self.suffix
    }

    pub fn is_compatible(&self, other: &Version) -> bool {
        other.get_major() <= self.major
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}.{}.{}{}", self.major, self.minor, self.patch, self.suffix)
    }
}

impl std::str::FromStr for Version {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = regex::Regex::new(r"(\d+)(\.(\d+))?(\.(\d+))?(.*)").unwrap();
        Ok(match re.captures(s) {
            Some(caps) => {
                Version {
                    major: caps.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    minor: caps.get(3).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    patch: caps.get(5).and_then(|m| m.as_str().parse().ok()).unwrap_or(0),
                    suffix: caps.get(6).map_or("", |m| m.as_str()).to_owned(),
                }
            }
            None => Version {
                major: 0,
                minor: 0,
                patch: 0,
                suffix: "".into(),
            }
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    pub(crate) fn test_read_doc_spec_block(
            block: &[u8],
            compression: compression::CompressionType,
    ) {
        let buff = Cursor::new(block);
        let data_attrs = DatasetAttributes {
            dimensions: vec![5, 6, 7],
            block_size: vec![1, 2, 3],
            data_type: DataType::INT16,
            compression: compression,
        };

        let block = <Foo as DefaultBlockReader<i16, std::io::Cursor<&[u8]>>>::read_block(
            buff,
            &data_attrs,
            vec![0, 0, 0]).expect("read_block failed");

        assert_eq!(block.get_size(), &vec![1, 2, 3]);
        assert_eq!(block.get_grid_position(), &vec![0, 0, 0]);
        assert_eq!(block.get_data(), &vec![1, 2, 3, 4, 5, 6]);
    }

    pub(crate) fn test_block_compression_rw(compression: compression::CompressionType) {
        let data_attrs = DatasetAttributes {
            dimensions: vec![10, 10, 10],
            block_size: vec![5, 5, 5],
            data_type: DataType::INT32,
            compression: compression,
        };
        let block_data: Vec<i32> = (0..125_i32).collect();
        let block_in = Box::new(VecDataBlock::new(
            data_attrs.block_size.clone(),
            vec![0, 0, 0],
            block_data.clone()));

        let mut inner: Vec<u8> = Vec::new();
        {
            let w_buff = Cursor::new(&mut inner);
            <Foo as DefaultBlockWriter<Vec<i32>, std::io::Cursor<_>>>::write_block(
                w_buff,
                &data_attrs,
                block_in).expect("write_block failed");
        }

        let block_out = <Foo as DefaultBlockReader<i32, _>>::read_block(
            &inner[..],
            &data_attrs,
            vec![0, 0, 0]).expect("read_block failed");

        assert_eq!(block_out.get_size(), &vec![5, 5, 5]);
        assert_eq!(block_out.get_grid_position(), &vec![0, 0, 0]);
        assert_eq!(block_out.get_data(), &block_data);
    }
}

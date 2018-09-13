//! Interfaces for the [N5 "Not HDF5" n-dimensional tensor file system storage
//! format](https://github.com/saalfeldlab/n5) created by the Saalfeld lab at
//! Janelia Research Campus.

extern crate byteorder;
#[cfg(feature = "bzip")]
extern crate bzip2;
#[cfg(feature = "gzip")]
extern crate flate2;
#[cfg(feature = "filesystem")]
extern crate fs2;
#[cfg(feature = "use_ndarray")]
extern crate itertools;
#[macro_use]
extern crate lazy_static;
#[cfg(feature = "lz")]
extern crate lz4;
#[cfg(feature = "use_ndarray")]
extern crate ndarray;
#[cfg(feature = "use_ndarray")]
extern crate num_traits;
extern crate serde;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
#[cfg(test)]
extern crate tempdir;
extern crate regex;
#[cfg(feature = "filesystem")]
extern crate walkdir;
#[cfg(feature = "xz")]
extern crate xz2;


use std::cmp;
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
#[cfg(feature = "use_ndarray")]
use itertools::Itertools;
use serde::Serialize;

use ::compression::Compression;

pub mod compression;
#[cfg(feature = "filesystem")]
pub mod filesystem;
pub mod prelude;


lazy_static! {
    static ref VERSION: Version = {
        Version::new(2, 0, 2, "")
    };
}

/// Key name for the version attribute in the container root.
const VERSION_ATTRIBUTE_KEY: &str = "n5";

/// Specifes the extents of an axis-aligned bounding box.
#[derive(Debug)]
pub struct BoundingBox {
    offset: Vec<i64>,
    size: Vec<i64>,
}

impl BoundingBox {
    pub fn new(offset: Vec<i64>, size: Vec<i64>) -> BoundingBox {
        BoundingBox {
            offset,
            size,
        }
    }
}

/// Non-mutating operations on N5 containers.
pub trait N5Reader {
    /// Get the N5 specification version of the container.
    fn get_version(&self) -> Result<Version, Error>;

    /// Get attributes for a dataset.
    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error>;

    /// Test whether a group or dataset exists.
    fn exists(&self, path_name: &str) -> bool;

    /// Test whether a dataset exists.
    fn dataset_exists(&self, path_name: &str) -> bool {
        self.exists(path_name) && self.get_dataset_attributes(path_name).is_ok()
    }

    /// Get a URI string for a data block.
    ///
    /// Whether this requires that the dataset and block exist is currently
    /// implementation dependent. Whether this URI is a URL is implementation
    /// dependent.
    fn get_block_uri(&self, path_name: &str, grid_position: &[i64]) -> Result<String, Error>;

    /// Read a single dataset block into a linear vec.
    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: Vec<i64>,
    ) -> Result<Option<VecDataBlock<T>>, Error>
        where DataType: DataBlockCreator<T>,
              VecDataBlock<T>: DataBlock<T>;

    /// Read an abitrary bounding box from an N5 volume in an ndarray, reading
    /// blocks in serial as necessary.
    ///
    /// Assumes blocks are column-major and returns a column-major ndarray.
    #[cfg(feature = "use_ndarray")]
    fn read_ndarray<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        bbox: &BoundingBox,
    ) -> Result<ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>, Error>
        where DataType: DataBlockCreator<T>,
              VecDataBlock<T>: DataBlock<T>,
              T: Clone + num_traits::identities::Zero {

        use ndarray::{
            Array,
            IxDyn,
            ShapeBuilder,
            SliceInfo,
        };

        // TODO: Verbose vector math and lots of allocations.
        let floor_coord = bbox.offset.iter()
            .zip(&data_attrs.block_size)
            .map(|(&o, &bs)| o / i64::from(bs));
        let ceil_coord = bbox.offset.iter()
            .zip(&bbox.size)
            .zip(&data_attrs.block_size)
            .map(|((&o, &s), &bs)| (o + s + i64::from(bs) - 1) / i64::from(bs));
        let offset = Array::from_vec(bbox.offset.clone());
        let size = Array::from_vec(bbox.size.clone());
        let arr_end = &offset + &size;
        let norm_block_size = Array::from_iter(data_attrs.block_size.iter().map(|n| i64::from(*n)));

        let coord_iter = floor_coord.zip(ceil_coord)
            .map(|(min, max)| min..max)
            .multi_cartesian_product();

        let arr_size: Vec<usize> = bbox.size.iter().map(|n| *n as usize).collect();
        let arr_size_a = Array::from_iter(arr_size.iter().map(|n| *n as i64));
        let mut arr = Array::zeros(arr_size.f());

        for coord in coord_iter {
            let block_opt = self.read_block(path_name, data_attrs, coord.clone())?;

            if let Some(block) = block_opt {
                let block_size = Array::from_iter(block.get_size().iter().map(|n| i64::from(*n)));
                let block_size_usize: Vec<usize> = block.get_size().iter().map(|n| *n as usize).collect();
                let coord_a = Array::from_vec(coord);

                let block_start = &coord_a * &norm_block_size;
                let block_end = &block_start + &block_size;
                let block_min = (&offset - &block_start).mapv(|v| cmp::max(v, 0));
                let block_max = block_size.clone() - (&block_end - &arr_end).mapv(|v| cmp::max(v, 0));

                let arr_start = (&block_start - &offset).mapv(|v| cmp::max(v, 0));
                let arr_end = arr_size_a.clone() - (&arr_end - &block_end).mapv(|v| cmp::max(v, 0));

                let arr_slice: Vec<ndarray::SliceOrIndex> = arr_start.iter().zip(&arr_end)
                    .map(|(&start, &end)| ndarray::SliceOrIndex::Slice {
                        start: start as isize,
                        end: Some(end as isize),
                        step: 1,
                    }).collect();
                let mut arr_view = arr.slice_mut(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

                let block_slice: Vec<ndarray::SliceOrIndex> = block_min.iter().zip(&block_max)
                    .map(|(&start, &end)| ndarray::SliceOrIndex::Slice {
                        start: start as isize,
                        end: Some(end as isize),
                        step: 1,
                    }).collect();

                // N5 datasets are stored f-order/column-major.
                let block_data = Array::from_shape_vec(block_size_usize.f(), block.into())
                    .expect("TODO: block ndarray failed");
                let block_view = block_data.slice(SliceInfo::<_, IxDyn>::new(block_slice).unwrap().as_ref());

                arr_view.assign(&block_view);
            }
        }

        Ok(arr)
    }

    /// List all groups (including datasets) in a group.
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error>;
}

/// Mutating operations on N5 containers.
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

    /// Create a dataset. This will create the dataset group and attributes,
    /// but not populate any block data.
    fn create_dataset(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
    ) -> Result<(), Error> {
        self.create_group(path_name)?;
        self.set_dataset_attributes(path_name, data_attrs)
    }

    /// Remove the N5 container.
    fn remove_all(&self) -> Result<(), Error> {
        self.remove("")
    }

    /// Remove a group or dataset (directory and all contained files).
    ///
    /// This will wait on locks acquired by other writers or readers.
    fn remove(
        &self,
        path_name: &str,
    ) -> Result<(), Error>;

    fn write_block<T, B: DataBlock<T>>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> Result<(), Error>;
}


/// Data types representable in N5.
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

impl DataType {
    /// Boilerplate method for reflection of primitive type sizes.
    pub fn size_of(self) -> usize {
        match self {
            DataType::UINT8 => std::mem::size_of::<u8>(),
            DataType::UINT16 => std::mem::size_of::<u16>(),
            DataType::UINT32 => std::mem::size_of::<u32>(),
            DataType::UINT64 => std::mem::size_of::<u64>(),
            DataType::INT8 => std::mem::size_of::<i8>(),
            DataType::INT16 => std::mem::size_of::<i16>(),
            DataType::INT32 => std::mem::size_of::<i32>(),
            DataType::INT64 => std::mem::size_of::<i64>(),
            DataType::FLOAT32 => std::mem::size_of::<f32>(),
            DataType::FLOAT64 => std::mem::size_of::<f64>(),
        }
    }
}

/// Reflect rust types to type values.
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
    ) -> Option<VecDataBlock<T>>;
}

macro_rules! data_type_block_creator {
    ($d_name:ident, $d_type:ty) => {
        impl TypeReflection<$d_type> for DataType {
            fn get_type_variant() -> DataType {
                DataType::$d_name
            }
        }

        impl DataBlockCreator<$d_type> for DataType {
            fn create_data_block(
                &self,
                block_size: Vec<i32>,
                grid_position: Vec<i64>,
                num_el: usize,
            ) -> Option<VecDataBlock<$d_type>> {
                match *self {
                    DataType::$d_name => Some(VecDataBlock::<$d_type>::new(
                        block_size,
                        grid_position,
                        vec![0. as $d_type; num_el],
                    )),
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

/// Attributes of a tensor dataset.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DatasetAttributes {
    /// Dimensions of the entire dataset, in voxels.
    dimensions: Vec<i64>,
    /// Size of each block, in voxels.
    block_size: Vec<i32>,
    /// Element data type.
    data_type: DataType,
    /// Compression scheme for voxel data in each block.
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
    fn read_data<R: std::io::Read>(&mut self, source: R) -> std::io::Result<()>;
}

pub trait WriteableDataBlock {
    fn write_data<W: std::io::Write>(&self, target: W) -> std::io::Result<()>;
}

/// Common interface for data blocks of element (rust) type `T`.
///
/// To enable custom types to be written to N5 volumes, implement this trait.
pub trait DataBlock<T> : Into<Vec<T>> + ReadableDataBlock + WriteableDataBlock {
    fn get_size(&self) -> &Vec<i32>;

    fn get_grid_position(&self) -> &Vec<i64>;

    fn get_data(&self) -> &Vec<T>;

    fn get_num_elements(&self) -> i32; // TODO: signed sizes feel awful.
}

/// A linear vector storing a data block. All read data blocks are returned as
/// this type.
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
            fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
                source.$bo_read_fn::<BigEndian>(&mut self.data)
            }
        }

        impl WriteableDataBlock for VecDataBlock<$ty_name> {
            fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
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
    fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
        source.read_exact(&mut self.data)
    }
}

impl WriteableDataBlock for VecDataBlock<u8> {
    fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
        target.write_all(&self.data)
    }
}

impl ReadableDataBlock for VecDataBlock<i8> {
    fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
        // Unsafe necessary here because we need a &mut [u8] to avoid doing
        // individual reads to the i8 data. This is safe.
        let data_ref = unsafe { &mut *(self.data.as_mut() as *mut [i8] as *mut [u8]) };
        source.read_exact(data_ref)
    }
}

impl WriteableDataBlock for VecDataBlock<i8> {
    fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
        // Unsafe necessary here because we need a &mut [u8] to avoid doing
        // individual writes from the i8 data. This is safe.
        let data_ref = unsafe { &*(self.data.as_ref() as *const [i8] as *const [u8]) };
        target.write_all(data_ref)
    }
}

impl<T> Into<Vec<T>> for VecDataBlock<T> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> DataBlock<T> for VecDataBlock<T>
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


/// Reads blocks from rust readers.
pub trait DefaultBlockReader<T, R: std::io::Read>
        where DataType: DataBlockCreator<T> {
    fn read_block(
        mut buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: Vec<i64>,
    ) -> std::io::Result<VecDataBlock<T>>
            where VecDataBlock<T>: DataBlock<T> {
        let mode = buffer.read_i16::<BigEndian>()?;
        let ndim = buffer.read_i16::<BigEndian>()?;
        let mut dims = vec![0; ndim as usize];
        buffer.read_i32_into::<BigEndian>(&mut dims)?;
        let num_el = match mode {
            0 => dims.iter().product(),
            1 => buffer.read_i32::<BigEndian>()?,
            _ => return Err(Error::new(ErrorKind::InvalidData, "Unsupported block mode"))
        };

        let mut block: VecDataBlock<T> = data_attrs.data_type.create_data_block(
            dims,
            grid_position,
            num_el as usize).expect("Attempt to create data block for wrong type.");
        let mut decompressed = data_attrs.compression.decoder(buffer);
        block.read_data(&mut decompressed)?;

        Ok(block)
    }
}

/// Writes blocks to rust writers.
pub trait DefaultBlockWriter<T, W: std::io::Write, B: DataBlock<T>> {
    fn write_block(
        mut buffer: W,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> std::io::Result<()> {
        let mode: i16 = if block.get_num_elements() == block.get_size().iter().product::<i32>()
            {0} else {1};
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
struct DefaultBlock;
impl<T, R: std::io::Read> DefaultBlockReader<T, R> for DefaultBlock
        where DataType: DataBlockCreator<T> {}
impl<T, W: std::io::Write, B: DataBlock<T>> DefaultBlockWriter<T, W, B> for DefaultBlock {}


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
            compression,
        };

        let block = <DefaultBlock as DefaultBlockReader<i16, std::io::Cursor<&[u8]>>>::read_block(
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
            compression,
        };
        let block_data: Vec<i32> = (0..125_i32).collect();
        let block_in = VecDataBlock::new(
            data_attrs.block_size.clone(),
            vec![0, 0, 0],
            block_data.clone());

        let mut inner: Vec<u8> = Vec::new();

        <DefaultBlock as DefaultBlockWriter<i32, _, _>>::write_block(
            &mut inner,
            &data_attrs,
            &block_in).expect("write_block failed");

        let block_out = <DefaultBlock as DefaultBlockReader<i32, _>>::read_block(
            &inner[..],
            &data_attrs,
            vec![0, 0, 0]).expect("read_block failed");

        assert_eq!(block_out.get_size(), &vec![5, 5, 5]);
        assert_eq!(block_out.get_grid_position(), &vec![0, 0, 0]);
        assert_eq!(block_out.get_data(), &block_data);
    }
}

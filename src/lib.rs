//! Interfaces for the [N5 "Not HDF5" n-dimensional tensor file system storage
//! format](https://github.com/saalfeldlab/n5) created by the Saalfeld lab at
//! Janelia Research Campus.


// TODO: this does not run the test for recent stable rust because `test`
// is no longer set during doc tests. When 1.40 stabilizes and is the MSRV
// this can be changed from `test` to `doctest` and will work correctly.
#[cfg(all(test, feature = "filesystem"))]
doc_comment::doctest!("../README.md");


#[macro_use]
pub extern crate smallvec;


use std::io::{
    Error,
    ErrorKind,
    Read,
    Write,
};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::time::SystemTime;

use byteorder::{
    BigEndian,
    ByteOrder,
    ReadBytesExt,
    WriteBytesExt,
};
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;
use smallvec::SmallVec;

use crate::compression::Compression;

pub mod compression;
#[macro_use]
pub mod data_type;
pub use data_type::*;
#[cfg(feature = "filesystem")]
pub mod filesystem;
#[cfg(feature = "use_ndarray")]
pub mod ndarray;
pub mod prelude;
pub mod version;

pub use crate::version::Version;


const COORD_SMALLVEC_SIZE: usize = 6;
pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;
pub type BlockCoord = CoordVec<u32>;
pub type GridCoord = CoordVec<u64>;


pub use version::VERSION;

/// Key name for the version attribute in the container root.
pub const VERSION_ATTRIBUTE_KEY: &str = "n5";

#[derive(Clone, Debug)]
pub struct DataBlockMetadata {
    pub created: SystemTime,
    pub accessed: SystemTime,
    pub modified: SystemTime,
}

pub trait ReadableStore {
    type GetReader: Read;

    /// TODO: not in zarr spec
    fn exists(&self, key: &str) -> Result<bool, Error>;

    fn get(&self, key: &str) -> Result<Option<Self::GetReader>, Error>;
}

pub trait WriteableStore {
    type SetWriter: Write;

    fn set<F: FnOnce(Self::SetWriter) -> Result<(), Error>>(&self, key: &str, value: F) -> Result<(), Error>;

    fn delete(&self, key: &str) -> Result<(), Error>;
}

/// Non-mutating operations on N5 containers.
pub trait N5Reader {
    /// Get the N5 specification version of the container.
    fn get_version(&self) -> Result<Version, Error>;

    /// Get attributes for a dataset.
    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error>;

    /// Test whether a group or dataset exists.
    fn exists(&self, path_name: &str) -> Result<bool, Error>;

    /// Test whether a dataset exists.
    fn dataset_exists(&self, path_name: &str) -> Result<bool, Error> {
        Ok(self.exists(path_name)? && self.get_dataset_attributes(path_name).is_ok())
    }

    /// Get a URI string for a data block.
    ///
    /// Whether this requires that the dataset and block exist is currently
    /// implementation dependent. Whether this URI is a URL is implementation
    /// dependent.
    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String, Error>;

    /// Read a single dataset block into a linear vec.
    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>, Error>
        where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
              T: ReflectedType;

    /// Read a single dataset block into an existing buffer.
    fn read_block_into<T: ReflectedType, B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>, Error>;

    /// Read metadata about a block.
    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>, Error>;

    /// List all attributes of a group.
    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error>;
}

impl<S: ReadableStore> N5Reader for S {
    fn get_version(&self) -> Result<Version, Error> {
        todo!()
    }

    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes, Error> {
        let dataset_path = PathBuf::from(path_name).join(ARRAY_METADATA_PATH);
        let value_reader = ReadableStore::get(self, &dataset_path.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(std::io::ErrorKind::NotFound))?;
        Ok(serde_json::from_reader(value_reader)?)
    }

    fn exists(&self, path_name: &str) -> Result<bool, Error> {
        self.exists(path_name)
    }

    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String, Error> {
        todo!()
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>, Error>
        where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
              T: ReflectedType {

        // TODO convert asserts to errors
        assert!(data_attrs.in_bounds(&grid_position));

        // Construct block path string
        let block_path = get_block_path(path_name, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &block_path)?;

        // Read value into container
        value_reader.map(|reader|
            <crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block(
                reader,
                data_attrs,
                grid_position))
            .transpose()
    }

    fn read_block_into<T: ReflectedType, B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>, Error> {
        // TODO convert asserts to errors
        assert!(data_attrs.in_bounds(&grid_position));

        // Construct block path string
        let block_path = get_block_path(path_name, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &block_path)?;

        // Read value into container
        value_reader.map(|reader|
            <crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block_into(
                reader,
                data_attrs,
                grid_position,
                block,
            ))
            .transpose()
    }

    fn block_metadata(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>, Error> {
        todo!()
    }

    fn list_attributes(&self, path_name: &str) -> Result<serde_json::Value, Error> {
        todo!()
    }
}

pub trait N5Lister {
    /// List all groups (including datasets) in a group.
    fn list(&self, path_name: &str) -> Result<Vec<String>, Error>;
}

fn get_block_path(base_path: &str, grid_position: &[u64]) -> String {
    use std::fmt::Write;
    // TODO cleanup
    let mut block_path = match grid_position.len() {
        0 => base_path.to_owned(),
        _ => format!("{}/", base_path),
    };
    write!(block_path, "{}",
        grid_position.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>().join(".")).unwrap();

    block_path
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
    ) -> Result<(), Error>;

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

    fn write_block<T, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &GridCoord,
        block: &B,
    ) -> Result<(), Error>;
}

// From: https://github.com/serde-rs/json/issues/377
// TODO: Could be much better.
fn merge(a: &mut Value, b: &Value) {
    match (a, b) {
        (&mut Value::Object(ref mut a), &Value::Object(ref b)) => {
            for (k, v) in b {
                merge(a.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (a, b) => {
            *a = b.clone();
        }
    }
}

impl<S: ReadableStore + WriteableStore> N5Writer for S {
    fn set_attributes(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), Error> {
        let path_buf = PathBuf::from(path_name);
        let metadata_path = if self.exists(path_buf.join(ARRAY_METADATA_PATH).to_str().expect("TODO"))? {
            path_buf.join(ARRAY_METADATA_PATH)
        } else if self.exists(path_buf.join(GROUP_METADATA_PATH).to_str().expect("TODO"))? {
            path_buf.join(GROUP_METADATA_PATH)
        } else {
            return Err(Error::new(std::io::ErrorKind::NotFound, "Node does not exist at path"))
        };

        // TODO: race condition
        let value_reader = ReadableStore::get(self, &metadata_path.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(std::io::ErrorKind::NotFound))?;
        let existing: Value = serde_json::from_reader(value_reader)?;
        
        let mut merged = existing.clone();
        let new: Value = attributes.into();
        merge(&mut merged, &new);
        if merged != existing {
            self.set(
                metadata_path.to_str().expect("TODO"),
                |writer| Ok(serde_json::to_writer(writer, &merged)?),
            )?;
        }
        Ok(())
    }

    fn create_group(&self, path_name: &str) -> Result<(), Error> {
        let path_buf = PathBuf::from(path_name);
        if let Some(parent) = path_buf.parent() {
            self.create_group(parent.to_str().expect("TODO"))?;
        }
        let metadata_path = path_buf.join(GROUP_METADATA_PATH);
        if self.exists(path_buf.join(ARRAY_METADATA_PATH).to_str().expect("TODO"))? {
            Err(Error::new(std::io::ErrorKind::AlreadyExists,
                "Array already exists at group path"))
        } else if self.exists(metadata_path.to_str().expect("TODO"))? {
            Ok(())
        } else {
            self.set(
                metadata_path.to_str().expect("TODO"),
                |writer| Ok(serde_json::to_writer(writer, &GroupMetadata::default())?),
            )
        }
    }

    fn create_dataset(&self, path_name: &str, data_attrs: &DatasetAttributes) -> Result<(), Error> {
        let path_buf = PathBuf::from(path_name);
        if let Some(parent) = path_buf.parent() {
            self.create_group(parent.to_str().expect("TODO"))?;
        }
        let metadata_path = path_buf.join(ARRAY_METADATA_PATH);
        if self.exists(path_buf.join(GROUP_METADATA_PATH).to_str().expect("TODO"))?
                || self.exists(metadata_path.to_str().expect("TODO"))? {
            Err(Error::new(std::io::ErrorKind::AlreadyExists,
                "Node already exists at array path"))
        } else {
            self.set(
                metadata_path.to_str().expect("TODO"),
                |writer| Ok(serde_json::to_writer(writer, data_attrs)?),
            )
        }
    }

    fn remove(
        &self,
        path_name: &str,
    ) -> Result<(), Error> {
        self.delete(path_name)
    }

    fn write_block<T, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: &GridCoord,
        block: &B,
    ) -> Result<(), Error> {

        // TODO convert assert
        // assert!(data_attrs.in_bounds(block.get_grid_position()));
        let block_path = get_block_path(path_name, grid_position);
        self.set(&block_path, |writer|
            <DefaultBlock as DefaultBlockWriter<T, _, _>>::write_block(
                writer, data_attrs, block,
            )
        )
    }
}

const ARRAY_METADATA_PATH: &str = ".array";
const GROUP_METADATA_PATH: &str = ".group";

/// Metadata for groups.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GroupMetadata {
    extensions: Vec<serde_json::Value>,
    attributes: serde_json::Map<String, serde_json::Value>
}

impl Default for GroupMetadata {
    fn default() -> Self {
        GroupMetadata {
            extensions: Vec::new(),
            attributes: serde_json::Map::new(),
        }
    }
}

/// Attributes of a tensor dataset.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DatasetAttributes {
    /// Dimensions of the entire dataset, in voxels.
    dimensions: GridCoord,
    /// Size of each block, in voxels.
    block_size: BlockCoord,
    /// Element data type.
    data_type: DataType,
    /// Compression scheme for voxel data in each block.
    compression: compression::CompressionType,
}

impl DatasetAttributes {
    pub fn new(
        dimensions: GridCoord,
        block_size: BlockCoord,
        data_type: DataType,
        compression: compression::CompressionType,
    ) -> DatasetAttributes {
        assert_eq!(dimensions.len(), block_size.len(),
            "Number of dataset dimensions must match number of block size dimensions.");
        DatasetAttributes {
            dimensions,
            block_size,
            data_type,
            compression,
        }
    }

    pub fn get_dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn get_block_size(&self) -> &[u32] {
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

    /// Get the total number of elements possible given the dimensions.
    pub fn get_num_elements(&self) -> usize {
        self.dimensions.iter().map(|&d| d as usize).product()
    }

    /// Get the total number of elements possible in a block.
    pub fn get_block_num_elements(&self) -> usize {
        self.block_size.iter().map(|&d| d as usize).product()
    }

    /// Get the upper bound extent of grid coordinates.
    pub fn get_grid_extent(&self) -> GridCoord {
        self.dimensions.iter()
            .zip(self.block_size.iter())
            .map(|(d, &b)| d / u64::from(b))
            .collect()
    }

    /// Check whether a block grid position is in the bounds of this dataset.
    /// ```
    /// use n5::prelude::*;
    /// use n5::smallvec::smallvec;
    /// let attrs = DatasetAttributes::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![10, 10, 10],
    ///     DataType::UINT8,
    ///     n5::compression::CompressionType::default(),
    /// );
    /// assert!(attrs.in_bounds(&smallvec![4, 3, 2]));
    /// assert!(!attrs.in_bounds(&smallvec![5, 3, 2]));
    /// ```
    pub fn in_bounds(&self, grid_position: &GridCoord) -> bool {
        self.dimensions.len() == grid_position.len() &&
        self.get_grid_extent().iter()
            .zip(grid_position.iter())
            .all(|(&bound, &coord)| coord < bound)
    }
}

pub trait ReinitDataBlock<T> {
    fn reinitialize(&mut self, num_el: usize);

    fn reinitialize_with<B: DataBlock<T>>(&mut self, other: &B);
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
pub trait DataBlock<T> {
    fn get_data(&self) -> &[T];

    fn get_num_elements(&self) -> u32;
}

/// A generic data block container wrapping any type that can be taken as a
/// slice ref.
#[derive(Clone)]
pub struct SliceDataBlock<T: ReflectedType, C> {
    data_type: PhantomData<T>,
    data: C,
}

/// A linear vector storing a data block. All read data blocks are returned as
/// this type.
pub type VecDataBlock<T> = SliceDataBlock<T, Vec<T>>;

impl<T: ReflectedType, C> SliceDataBlock<T, C> {
    pub fn new(data: C) -> SliceDataBlock<T, C> {
        SliceDataBlock {
            data_type: PhantomData,
            data,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

impl<T: ReflectedType> ReinitDataBlock<T> for VecDataBlock<T> {
    fn reinitialize(&mut self, num_el: usize) {
        self.data.resize_with(num_el, Default::default);
    }

    fn reinitialize_with<B: DataBlock<T>>(&mut self, other: &B) {
        self.data.clear();
        self.data.extend_from_slice(other.get_data());
    }
}

macro_rules! vec_data_block_impl {
    ($ty_name:ty, $bo_read_fn:ident, $bo_write_fn:ident) => {
        impl<C: AsMut<[$ty_name]>> ReadableDataBlock for SliceDataBlock<$ty_name, C> {
            fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
                source.$bo_read_fn::<BigEndian>(self.data.as_mut())
            }
        }

        impl<C: AsRef<[$ty_name]>> WriteableDataBlock for SliceDataBlock<$ty_name, C> {
            fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
                const CHUNK: usize = 256;
                let mut buf: [u8; CHUNK * std::mem::size_of::<$ty_name>()] =
                    [0; CHUNK * std::mem::size_of::<$ty_name>()];

                for c in self.data.as_ref().chunks(CHUNK) {
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

impl<C: AsMut<[u8]>> ReadableDataBlock for SliceDataBlock<u8, C> {
    fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
        source.read_exact(self.data.as_mut())
    }
}

impl<C: AsRef<[u8]>> WriteableDataBlock for SliceDataBlock<u8, C> {
    fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
        target.write_all(self.data.as_ref())
    }
}

impl<C: AsMut<[i8]>> ReadableDataBlock for SliceDataBlock<i8, C> {
    fn read_data<R: std::io::Read>(&mut self, mut source: R) -> std::io::Result<()> {
        // Unsafe necessary here because we need a &mut [u8] to avoid doing
        // individual reads to the i8 data. This is safe.
        // Note that byteorder's read_i8_into is not used, because it is also
        // unsafe under the hood and moreso than this incantation.
        let data_ref = unsafe { &mut *(self.data.as_mut() as *mut [i8] as *mut [u8]) };
        source.read_exact(data_ref)
    }
}

impl<C: AsRef<[i8]>> WriteableDataBlock for SliceDataBlock<i8, C> {
    fn write_data<W: std::io::Write>(&self, mut target: W) -> std::io::Result<()> {
        // Unsafe necessary here because we need a &mut [u8] to avoid doing
        // individual writes from the i8 data. This is safe.
        let data_ref = unsafe { &*(self.data.as_ref() as *const [i8] as *const [u8]) };
        target.write_all(data_ref)
    }
}

impl<T: ReflectedType, C: AsRef<[T]>> DataBlock<T> for SliceDataBlock<T, C> {
    fn get_data(&self) -> &[T] {
        self.data.as_ref()
    }

    fn get_num_elements(&self) -> u32 {
        self.data.as_ref().len() as u32
    }
}


/// Reads blocks from rust readers.
pub trait DefaultBlockReader<T: ReflectedType, R: std::io::Read> {
    fn read_block(
        mut buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> std::io::Result<VecDataBlock<T>>
            where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock {

        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type."))
        }
    
        let mut block = T::create_data_block(data_attrs.get_block_num_elements());
        let mut decompressed = data_attrs.compression.decoder(buffer);
        block.read_data(&mut decompressed)?;

        Ok(block)
    }

    fn read_block_into<B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock>(
        mut buffer: R,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> std::io::Result<()> {

        if data_attrs.data_type != T::VARIANT {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Attempt to create data block for wrong type."))
        }

        block.reinitialize(data_attrs.get_block_num_elements());
        let mut decompressed = data_attrs.compression.decoder(buffer);
        block.read_data(&mut decompressed)?;

        Ok(())
    }
}

/// Writes blocks to rust writers.
pub trait DefaultBlockWriter<T, W: std::io::Write, B: DataBlock<T> + WriteableDataBlock> {
    fn write_block(
        mut buffer: W,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> std::io::Result<()> {
        let expected_len = data_attrs.get_block_num_elements();
        let actual_len = block.get_num_elements();
        if expected_len != actual_len as usize {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Block is wrong size"))
        } else {
            let mut compressor = data_attrs.compression.encoder(buffer);
            block.write_data(&mut compressor)?;

            Ok(())
        }
    }
}

// TODO: needed because cannot invoke type parameterized static trait methods
// directly from trait name in Rust. Symptom of design problems with
// `DefaultBlockReader`, etc.
pub struct DefaultBlock;
impl<T: ReflectedType, R: std::io::Read> DefaultBlockReader<T, R> for DefaultBlock {}
impl<T, W: std::io::Write, B: DataBlock<T> + WriteableDataBlock> DefaultBlockWriter<T, W, B> for DefaultBlock {}


#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    const DOC_SPEC_BLOCK_DATA: [i16; 6] = [1, 2, 3, 4, 5, 6];

    fn doc_spec_dataset_attributes(compression: compression::CompressionType) -> DatasetAttributes {
        DatasetAttributes {
            dimensions: smallvec![5, 6, 7],
            block_size: smallvec![1, 2, 3],
            data_type: DataType::INT16,
            compression,
        }
    }

    pub(crate) fn test_block_compression_rw(compression: compression::CompressionType) {
        let data_attrs = DatasetAttributes {
            dimensions: smallvec![10, 10, 10],
            block_size: smallvec![5, 5, 5],
            data_type: DataType::INT32,
            compression,
        };
        let block_data: Vec<i32> = (0..125_i32).collect();
        let block_in = SliceDataBlock::new(&block_data);

        let mut inner: Vec<u8> = Vec::new();

        <DefaultBlock as DefaultBlockWriter<i32, _, _>>::write_block(
            &mut inner,
            &data_attrs,
            &block_in).expect("write_block failed");

        let block_out = <DefaultBlock as DefaultBlockReader<i32, _>>::read_block(
            &inner[..],
            &data_attrs,
            smallvec![0, 0, 0]).expect("read_block failed");

        assert_eq!(block_out.get_num_elements(), 5*5*5);
        assert_eq!(block_out.get_data(), &block_data[..]);
    }

}

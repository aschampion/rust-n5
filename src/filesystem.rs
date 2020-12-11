//! A filesystem-backed N5 container.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Error, ErrorKind, Read, Result, Seek, SeekFrom};
use std::path::PathBuf;
use std::str::FromStr;

use fs2::FileExt;
use serde_json::{self, json, Value};
use walkdir::WalkDir;

use crate::{
    is_version_compatible, DataBlock, DataBlockMetadata, DatasetAttributes, DefaultBlockReader,
    DefaultBlockWriter, GridCoord, N5Lister, N5Reader, N5Writer, ReadableDataBlock, ReflectedType,
    ReinitDataBlock, VecDataBlock, Version, WriteableDataBlock,
};

/// Name of the attributes file stored in the container root and dataset dirs.
const ATTRIBUTES_FILE: &str = "attributes.json";

/// A filesystem-backed N5 container.
#[derive(Clone, Debug)]
pub struct N5Filesystem {
    base_path: PathBuf,
}

impl N5Filesystem {
    /// Open an existing N5 container by path.
    pub fn open<P: AsRef<std::path::Path>>(base_path: P) -> Result<N5Filesystem> {
        let reader = N5Filesystem {
            base_path: PathBuf::from(base_path.as_ref()),
        };

        if reader.exists("")? {
            let version = reader.get_version()?;

            if !is_version_compatible(&crate::VERSION, &version) {
                return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"));
            }
        }

        Ok(reader)
    }

    /// Open an existing N5 container by path or create one if none exists.
    ///
    /// Note this will update the version attribute for existing containers.
    pub fn open_or_create<P: AsRef<std::path::Path>>(base_path: P) -> Result<N5Filesystem> {
        let reader = N5Filesystem {
            base_path: PathBuf::from(base_path.as_ref()),
        };

        fs::create_dir_all(base_path)?;

        if reader
            .get_version()
            .map(|v| !is_version_compatible(&crate::VERSION, &v))
            .unwrap_or(false)
        {
            return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"));
        } else {
            reader.set_attribute(
                "",
                crate::VERSION_ATTRIBUTE_KEY.to_owned(),
                crate::VERSION.to_string(),
            )?;
        }

        Ok(reader)
    }

    pub fn get_attributes(&self, path_name: &str) -> Result<Value> {
        let path = self.get_path(path_name)?;
        if path.is_dir() {
            let attr_path = path.join(ATTRIBUTES_FILE);

            if attr_path.exists() && attr_path.is_file() {
                let file = File::open(attr_path)?;
                file.lock_shared()?;
                let reader = BufReader::new(file);
                Ok(serde_json::from_reader(reader)?)
            } else {
                Ok(json!({}))
            }
        } else {
            Err(Error::new(ErrorKind::NotFound, "Path does not exist"))
        }
    }

    /// Get the filesystem path for a given N5 data path.
    fn get_path(&self, path_name: &str) -> Result<PathBuf> {
        // Note: cannot use `canonicalize` on both the constructed dataset path
        // and `base_path` and check `starts_with`, because `canonicalize` also
        // requires the path exist.
        use std::path::{Component, Path};

        // Normalize the path to be relative.
        let mut components = Path::new(path_name).components();
        while components.as_path().has_root() {
            match components.next() {
                Some(Component::Prefix(_)) => {
                    return Err(Error::new(
                        ErrorKind::NotFound,
                        "Path name is outside this N5 filesystem on a prefix path",
                    ))
                }
                Some(Component::RootDir) => (),
                // This should be unreachable.
                _ => return Err(Error::new(ErrorKind::NotFound, "Path is malformed")),
            }
        }
        let unrooted_path = components.as_path();

        // Check that the path is inside the container's base path.
        let mut nest: i32 = 0;
        for component in unrooted_path.components() {
            match component {
                // This should be unreachable.
                Component::Prefix(_) | Component::RootDir => {
                    return Err(Error::new(ErrorKind::NotFound, "Path is malformed"))
                }
                Component::CurDir => continue,
                Component::ParentDir => nest -= 1,
                Component::Normal(_) => nest += 1,
            };
        }

        if nest < 0 {
            Err(Error::new(
                ErrorKind::NotFound,
                "Path name is outside this N5 filesystem",
            ))
        } else {
            Ok(self.base_path.join(unrooted_path))
        }
    }

    fn get_data_block_path(&self, path_name: &str, grid_position: &[u64]) -> Result<PathBuf> {
        let mut path = self.get_path(path_name)?;
        for coord in grid_position {
            path.push(coord.to_string());
        }
        Ok(path)
    }

    fn get_attributes_path(&self, path_name: &str) -> Result<PathBuf> {
        let mut path = self.get_path(path_name)?;
        path.push(ATTRIBUTES_FILE);
        Ok(path)
    }
}

impl N5Reader for N5Filesystem {
    fn get_version(&self) -> Result<Version> {
        // TODO: dedicated error type should clean this up.
        Ok(Version::from_str(
            self.get_attributes("")?
                .get(crate::VERSION_ATTRIBUTE_KEY)
                .ok_or_else(|| Error::new(ErrorKind::NotFound, "Version attribute not present"))?
                .as_str()
                .unwrap_or(""),
        )
        .unwrap())
    }

    fn get_dataset_attributes(&self, path_name: &str) -> Result<DatasetAttributes> {
        let attr_path = self.get_attributes_path(path_name)?;
        let reader = BufReader::new(File::open(attr_path)?);
        Ok(serde_json::from_reader(reader)?)
    }

    fn exists(&self, path_name: &str) -> Result<bool> {
        let target = self.get_path(path_name)?;
        Ok(target.is_dir())
    }

    fn get_block_uri(&self, path_name: &str, grid_position: &[u64]) -> Result<String> {
        self.get_data_block_path(path_name, grid_position)?
            .to_str()
            // TODO: could use URL crate and `from_file_path` here.
            .map(|s| format!("file://{}", s))
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Paths must be UTF-8"))
    }

    fn read_block<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataBlock<T>>>
    where
        VecDataBlock<T>: DataBlock<T> + ReadableDataBlock,
        T: ReflectedType,
    {
        let block_file = self.get_data_block_path(path_name, &grid_position)?;
        if block_file.is_file() {
            let file = File::open(block_file)?;
            file.lock_shared()?;
            let reader = BufReader::new(file);
            Ok(Some(
                <crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block(
                    reader,
                    data_attrs,
                    grid_position,
                )?,
            ))
        } else {
            Ok(None)
        }
    }

    fn read_block_into<
        T: ReflectedType,
        B: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
    >(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        grid_position: GridCoord,
        block: &mut B,
    ) -> Result<Option<()>> {
        let block_file = self.get_data_block_path(path_name, &grid_position)?;
        if block_file.is_file() {
            let file = File::open(block_file)?;
            file.lock_shared()?;
            let reader = BufReader::new(file);
            <crate::DefaultBlock as DefaultBlockReader<T, _>>::read_block_into(
                reader,
                data_attrs,
                grid_position,
                block,
            )?;
            Ok(Some(()))
        } else {
            Ok(None)
        }
    }

    fn block_metadata(
        &self,
        path_name: &str,
        _data_attrs: &DatasetAttributes,
        grid_position: &[u64],
    ) -> Result<Option<DataBlockMetadata>> {
        let block_file = self.get_data_block_path(path_name, grid_position)?;
        if block_file.is_file() {
            let metadata = std::fs::metadata(block_file)?;
            Ok(Some(DataBlockMetadata {
                created: metadata.created().ok(),
                accessed: metadata.accessed().ok(),
                modified: metadata.modified().ok(),
                size: Some(metadata.len()),
            }))
        } else {
            Ok(None)
        }
    }

    // TODO: dupe with get_attributes w/ different empty behaviors
    fn list_attributes(&self, path_name: &str) -> Result<Value> {
        let attr_path = self.get_attributes_path(path_name)?;
        let file = File::open(attr_path)?;
        file.lock_shared()?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }
}

impl N5Lister for N5Filesystem {
    fn list(&self, path_name: &str) -> Result<Vec<String>> {
        // TODO: shouldn't do this in a closure to not equivocate errors with Nones.
        Ok(fs::read_dir(self.get_path(path_name)?)?
            .filter_map(|e| {
                if let Ok(file) = e {
                    if fs::metadata(file.path())
                        .map(|f| f.file_type().is_dir())
                        .ok()
                        == Some(true)
                    {
                        file.file_name().into_string().ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect())
    }
}

fn merge_top_level(a: &mut Value, b: serde_json::Map<String, Value>) {
    match a {
        &mut Value::Object(ref mut a) => {
            for (k, v) in b {
                a.insert(k, v);
            }
        }
        a => {
            *a = b.into();
        }
    }
}

impl N5Writer for N5Filesystem {
    fn set_attributes(
        &self,
        path_name: &str,
        attributes: serde_json::Map<String, Value>,
    ) -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(self.get_attributes_path(path_name)?)?;
        file.lock_exclusive()?;

        let mut existing_buf = String::new();
        file.read_to_string(&mut existing_buf)?;
        let existing = serde_json::from_str(&existing_buf).unwrap_or_else(|_| json!({}));
        let mut merged = existing.clone();

        merge_top_level(&mut merged, attributes);

        if merged != existing {
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &merged)?;
        }

        Ok(())
    }

    fn create_group(&self, path_name: &str) -> Result<()> {
        let path = self.get_path(path_name)?;
        fs::create_dir_all(path)
    }

    fn remove(&self, path_name: &str) -> Result<()> {
        let path = self.get_path(path_name)?;

        for entry in WalkDir::new(path).contents_first(true) {
            let entry = entry?;

            if entry.file_type().is_dir() {
                fs::remove_dir(entry.path())?;
            } else {
                let file = File::open(entry.path())?;
                file.lock_exclusive()?;
                fs::remove_file(entry.path())?;
            }
        }

        Ok(())
    }

    fn write_block<T: ReflectedType, B: DataBlock<T> + WriteableDataBlock>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        block: &B,
    ) -> Result<()> {
        let path = self.get_data_block_path(path_name, block.get_grid_position())?;
        fs::create_dir_all(path.parent().expect("TODO: root block path?"))?;

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        file.lock_exclusive()?;
        // Truncate after the lock is acquired, rather than on opening.
        file.set_len(0)?;

        let buffer = BufWriter::new(file);
        <crate::DefaultBlock as DefaultBlockWriter<T, _, _>>::write_block(buffer, data_attrs, block)
    }

    fn delete_block(&self, path_name: &str, grid_position: &[u64]) -> Result<bool> {
        let path = self.get_data_block_path(path_name, grid_position)?;

        if path.exists() {
            let file = fs::OpenOptions::new().read(true).open(&path)?;
            file.lock_exclusive()?;
            fs::remove_file(&path)?;
        }

        Ok(!path.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend;
    use crate::tests::{ContextWrapper, N5Testable};
    use tempdir::TempDir;

    impl crate::tests::N5Testable for N5Filesystem {
        type Wrapper = ContextWrapper<TempDir, N5Filesystem>;

        fn temp_new_rw() -> Self::Wrapper {
            let dir = TempDir::new("rust_n5_tests").unwrap();
            let n5 =
                N5Filesystem::open_or_create(dir.path()).expect("Failed to create N5 filesystem");

            ContextWrapper { context: dir, n5 }
        }

        fn open_reader(&self) -> Self {
            N5Filesystem::open(&self.base_path).unwrap()
        }
    }

    test_backend!(N5Filesystem);

    #[test]
    fn reject_exterior_paths() {
        let wrapper = N5Filesystem::temp_new_rw();
        let create = wrapper.as_ref();

        assert!(create.get_path("/").is_ok());
        assert_eq!(create.get_path("/").unwrap(), create.get_path("").unwrap());
        assert!(create.get_path("/foo/bar").is_ok());
        assert_eq!(
            create.get_path("/foo/bar").unwrap(),
            create.get_path("foo/bar").unwrap()
        );
        assert!(create.get_path("//").is_ok());
        assert_eq!(create.get_path("//").unwrap(), create.get_path("").unwrap());
        assert!(create.get_path("/..").is_err());
        assert!(create.get_path("..").is_err());
        assert!(create.get_path("foo/bar/baz/../../..").is_ok());
        assert!(create.get_path("foo/bar/baz/../../../..").is_err());
    }

    #[test]
    fn accept_hardlink_attributes() {
        let wrapper = N5Filesystem::temp_new_rw();
        let dir = TempDir::new("rust_n5_tests_dupe").unwrap();
        let mut attr_path = dir.path().to_path_buf();
        attr_path.push(ATTRIBUTES_FILE);

        std::fs::hard_link(wrapper.n5.get_attributes_path("").unwrap(), &attr_path).unwrap();

        wrapper.n5.set_attribute("", "foo".into(), "bar").unwrap();

        let dupe = N5Filesystem::open(dir.path()).unwrap();
        assert_eq!(dupe.get_attributes("").unwrap()["foo"], "bar");
    }

    #[test]
    fn list_symlinked_datasets() {
        let wrapper = N5Filesystem::temp_new_rw();
        let dir = TempDir::new("rust_n5_tests_dupe").unwrap();
        let mut linked_path = wrapper.context.path().to_path_buf();
        linked_path.push("linked_dataset");

        #[cfg(target_family = "unix")]
        std::os::unix::fs::symlink(dir.path(), &linked_path).unwrap();
        #[cfg(target_family = "windows")]
        std::os::windows::fs::symlink_dir(dir.path(), &linked_path).unwrap();

        assert_eq!(wrapper.n5.list("").unwrap(), vec!["linked_dataset"]);
        assert!(wrapper.n5.exists("linked_dataset").unwrap());

        let data_attrs = DatasetAttributes::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            crate::DataType::INT32,
            crate::compression::CompressionType::Raw(
                crate::compression::raw::RawCompression::default(),
            ),
        );
        wrapper
            .n5
            .create_dataset("linked_dataset", &data_attrs)
            .expect("Failed to create dataset");
        assert!(wrapper.n5.dataset_exists("linked_dataset").unwrap());
    }

    #[test]
    fn test_get_block_uri() {
        let dir = TempDir::new("rust_n5_tests").unwrap();
        let path_str = dir.path().to_str().unwrap();

        let create =
            N5Filesystem::open_or_create(path_str).expect("Failed to create N5 filesystem");
        let uri = create.get_block_uri("foo/bar", &vec![1, 2, 3]).unwrap();
        assert_eq!(uri, format!("file://{}/foo/bar/1/2/3", path_str));
    }

    #[test]
    pub(crate) fn short_block_truncation() {
        let wrapper = N5Filesystem::temp_new_rw();
        let create = wrapper.as_ref();
        let data_attrs = DatasetAttributes::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            crate::DataType::INT32,
            crate::compression::CompressionType::Raw(
                crate::compression::raw::RawCompression::default(),
            ),
        );
        let block_data: Vec<i32> = (0..125_i32).collect();
        let block_in = crate::SliceDataBlock::new(
            data_attrs.block_size.clone(),
            smallvec![0, 0, 0],
            &block_data,
        );

        create
            .create_dataset("foo/bar", &data_attrs)
            .expect("Failed to create dataset");
        create
            .write_block("foo/bar", &data_attrs, &block_in)
            .expect("Failed to write block");

        let read = create.open_reader();
        let block_out = read
            .read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 0])
            .expect("Failed to read block")
            .expect("Block is empty");
        let missing_block_out = read
            .read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 1])
            .expect("Failed to read block");

        assert_eq!(block_out.get_data(), &block_data[..]);
        assert!(missing_block_out.is_none());

        // Shorten data (this still will not catch trailing data less than the length).
        let block_data: Vec<i32> = (0..10_i32).collect();
        let block_in = crate::SliceDataBlock::new(
            data_attrs.block_size.clone(),
            smallvec![0, 0, 0],
            &block_data,
        );
        create
            .write_block("foo/bar", &data_attrs, &block_in)
            .expect("Failed to write block");

        let block_file = create.get_data_block_path("foo/bar", &[0, 0, 0]).unwrap();
        let file = File::open(block_file).unwrap();
        let metadata = file.metadata().unwrap();

        let header_len = 2 * std::mem::size_of::<u16>() + 4 * std::mem::size_of::<u32>();
        assert_eq!(
            metadata.len(),
            (header_len + block_data.len() * std::mem::size_of::<i32>()) as u64
        );
    }
}

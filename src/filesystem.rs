//! A filesystem-backed N5 container.

use std::fs::{
    self,
    File,
};
use std::io::{
    Error,
    ErrorKind,
    BufReader,
    BufWriter,
    Read,
    Result,
    Seek,
    SeekFrom,
};
use std::path::{
    PathBuf,
};
use std::str::FromStr;

use fs2::FileExt;
use serde_json::{
    self,
    json,
    Value,
};
use walkdir::WalkDir;

use crate::{
    DataBlock,
    DataBlockMetadata,
    DatasetAttributes,
    DefaultBlockReader,
    DefaultBlockWriter,
    GridCoord,
    ReadableDataBlock,
    ReadableStore,
    ReflectedType,
    ReinitDataBlock,
    VecDataBlock,
    Version,
    WriteableDataBlock,
    WriteableStore,
};


/// Name of the attributes file stored in the container root and dataset dirs.
const ATTRIBUTES_FILE: &str = "attributes.json";


/// A filesystem-backed N5 container.
#[derive(Clone)]
pub struct N5Filesystem {
    base_path: PathBuf,
}

impl N5Filesystem {
    /// Open an existing N5 container by path.
    pub fn open(base_path: &str) -> Result<N5Filesystem> {
        let reader = N5Filesystem {
            base_path: PathBuf::from(base_path),
        };

        // if reader.exists("")? {
        //     let version = reader.get_version()?;

        //     if !crate::VERSION.is_compatible(&version) {
        //         return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"))
        //     }
        // }

        Ok(reader)
    }

    /// Open an existing N5 container by path or create one if none exists.
    ///
    /// Note this will update the version attribute for existing containers.
    pub fn open_or_create(base_path: &str) -> Result<N5Filesystem> {
        let reader = N5Filesystem {
            base_path: PathBuf::from(base_path),
        };

        fs::create_dir_all(base_path)?;

        // if reader.get_version().map(|v| !v.is_compatible(&crate::VERSION)).unwrap_or(false) {
        //     return Err(Error::new(ErrorKind::Other, "TODO: Incompatible version"))
        // } else {
        //     reader.set_attribute("", crate::VERSION_ATTRIBUTE_KEY.to_owned(), crate::VERSION.to_string())?;
        // }

        Ok(reader)
    }

    pub fn get_attributes(&self, path_name: &str) -> Result<Value> {
        if self.exists(path_name)? {
            let attr_path = self.base_path.join(path_name).join(ATTRIBUTES_FILE);

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

    fn get_path(&self, path_name: &str) -> Result<PathBuf> {
        // Note: cannot use `canonicalize` on both the constructed dataset path
        // and `base_path` and check `starts_with`, because `canonicalize` also
        // requires the path exist.
        use std::path::Component;

        // TODO: cleanup?
        let data_path = PathBuf::from(path_name);
        if data_path.is_relative() {
            let mut nest: i32 = 0;
            let mut interior = true;
            for component in data_path.components() {
                match component {
                    Component::Prefix(_) => unreachable!(), // Not an absolute path.
                    Component::RootDir => unreachable!(), // Not an absolute path.
                    Component::CurDir => continue,
                    Component::ParentDir => nest -= 1,
                    Component::Normal(_) => nest += 1,
                };

                if nest < 0 {
                    interior = false
                }
            }

            if interior {
                return Ok(self.base_path.join(path_name))
            }
        }

        Err(Error::new(ErrorKind::NotFound, "Path name is outside this N5 filesystem"))
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

impl ReadableStore for N5Filesystem {
    type GetReader = BufReader<File>;

    fn exists(&self, key: &str) -> Result<bool> {
        let target = self.base_path.join(key);
        Ok(target.is_file())
    }

    fn get(&self, key: &str) -> Result<Option<Self::GetReader>> {
        let target = self.base_path.join(key);
        if target.is_file() {
            let file = File::open(target)?;
            // TODO: lock
            Ok(Some(BufReader::new(file)))
        } else {
            Ok(None)
        }
    }
}

impl WriteableStore for N5Filesystem {
    type SetWriter = BufWriter<File>;

    fn set<F: FnOnce(Self::SetWriter) -> Result<()>>(&self, key: &str, value: F) -> Result<()> {
        let target = self.base_path.join(key);
        if let Some(parent) = target.parent() {
            if !parent.exists() {
                fs::create_dir(parent)?;
            }
        }
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(target)?;
        
        let writer = BufWriter::new(file);
        
        value(writer)
    }

    fn delete(&self, key: &str) -> Result<()> {
        todo!();
    }

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




#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataType,
        N5Reader,
        N5Writer,
    };
    use tempdir::TempDir;

    #[test]
    fn create_filesystem() {
        let dir = TempDir::new("rust_n5_tests").unwrap();
        let path_str = dir.path().to_str().unwrap();

        let create = N5Filesystem::open_or_create(path_str)
            .expect("Failed to create N5 filesystem");
        // create.set_attribute("", "foo".to_owned(), "bar")
        //     .expect("Failed to set attribute");

        let read = N5Filesystem::open(path_str)
            .expect("Failed to open N5 filesystem");

        // assert_eq!(read.get_version().expect("Cannot read version"), crate::VERSION);
        // assert_eq!(read.list_attributes("").unwrap()["foo"], "bar");
    }

    #[test]
    fn create_dataset() {
        let dir = TempDir::new("rust_n5_tests").unwrap();
        let path_str = dir.path().to_str().unwrap();

        let create = N5Filesystem::open_or_create(path_str)
            .expect("Failed to create N5 filesystem");
        let data_attrs = DatasetAttributes::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            DataType::INT32,
            crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
        );
        create.create_dataset("foo/bar", &data_attrs)
            .expect("Failed to create dataset");

        let read = N5Filesystem::open(path_str)
            .expect("Failed to open N5 filesystem");

        // assert_eq!(read.get_dataset_attributes("foo/bar").unwrap(), data_attrs);
    }

    #[test]
    fn reject_exterior_paths() {
        let dir = TempDir::new("rust_n5_tests").unwrap();
        let path_str = dir.path().to_str().unwrap();

        let create = N5Filesystem::open_or_create(path_str)
            .expect("Failed to create N5 filesystem");

        assert!(create.get_path("/").is_err());
        assert!(create.get_path("..").is_err());
        assert!(create.get_path("foo/bar/baz/../../..").is_ok());
        assert!(create.get_path("foo/bar/baz/../../../..").is_err());
    }

    // #[test]
    // fn test_get_block_uri() {
    //     let dir = TempDir::new("rust_n5_tests").unwrap();
    //     let path_str = dir.path().to_str().unwrap();

    //     let create = N5Filesystem::open_or_create(path_str)
    //         .expect("Failed to create N5 filesystem");
    //     let uri = create.get_block_uri("foo/bar", &vec![1, 2, 3]).unwrap();
    //     assert_eq!(uri, format!("file://{}/foo/bar/1/2/3", path_str));
    // }

    #[test]
    fn create_block_rw() {
        let dir = TempDir::new("rust_n5_tests").unwrap();
        let path_str = dir.path().to_str().unwrap();

        let create = N5Filesystem::open_or_create(path_str)
            .expect("Failed to create N5 filesystem");
        let data_attrs = DatasetAttributes::new(
            smallvec![10, 10, 10],
            smallvec![5, 5, 5],
            DataType::INT32,
            crate::compression::CompressionType::Raw(crate::compression::raw::RawCompression::default()),
        );
        let block_data: Vec<i32> = (0..125_i32).collect();
        let block_in = crate::SliceDataBlock::new(
            data_attrs.block_size.clone(),
            smallvec![0, 0, 0],
            &block_data);

        create.create_dataset("foo/bar", &data_attrs)
            .expect("Failed to create dataset");
        create.write_block("foo/bar", &data_attrs, &block_in)
            .expect("Failed to write block");

        let read = N5Filesystem::open(path_str)
            .expect("Failed to open N5 filesystem");
        let block_out = read.read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 0])
            .expect("Failed to read block")
            .expect("Block is empty");
        let missing_block_out = read.read_block::<i32>("foo/bar", &data_attrs, smallvec![0, 0, 1])
            .expect("Failed to read block");

        assert_eq!(block_out.get_data(), &block_data[..]);
        assert!(missing_block_out.is_none());
    }
}

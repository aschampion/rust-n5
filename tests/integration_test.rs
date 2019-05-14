extern crate n5;
extern crate rand;
#[macro_use]
extern crate smallvec;
extern crate tempdir;


use rand::Rng;

use n5::prelude::*;


fn test_read_write<T, N5: N5Reader + N5Writer>(
        n: &N5,
        compression: &CompressionType,
        dim: usize,
) where T: 'static + std::fmt::Debug + Clone + PartialEq,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
        DataType: TypeReflection<T>,
        VecDataBlock<T>: n5::ReadableDataBlock + n5::WriteableDataBlock,
        DataType: n5::DataBlockCreator<T> {
    let block_size: BlockCoord = (1..=dim as i32).rev().map(|d| d*5).collect();
    let data_attrs = DatasetAttributes::new(
        (1..=dim as i64).map(|d| d*100).collect(),
        block_size.clone(),
        <DataType as TypeReflection<T>>::get_type_variant(),
        compression.clone(),
    );
    let numel = data_attrs.get_block_num_elements();
    let mut rng = rand::thread_rng();
    let mut block_data = Vec::<T>::with_capacity(numel);

    for _ in 0..numel {
        block_data.push(rng.gen());
    }

    let block_in = VecDataBlock::new(
        block_size,
        smallvec![0; dim],
        block_data.clone());

    let path_name = "test/dataset/group";

    n.create_dataset(path_name, &data_attrs)
        .expect("Failed to create dataset");
    n.write_block(path_name, &data_attrs, &block_in)
        .expect("Failed to write block");

    let block_out = n.read_block::<T>(path_name, &data_attrs, smallvec![0; dim])
        .expect("Failed to read block")
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data);

    n.remove(path_name).unwrap();
}

fn test_all_types<N5: N5Reader + N5Writer>(
        n: &N5,
        compression: &CompressionType,
        dim: usize,
) {
    test_read_write::<u8, _>(n, compression, dim);
    test_read_write::<u16, _>(n, compression, dim);
    test_read_write::<u32, _>(n, compression, dim);
    test_read_write::<u64, _>(n, compression, dim);
    test_read_write::<i8, _>(n, compression, dim);
    test_read_write::<i16, _>(n, compression, dim);
    test_read_write::<i32, _>(n, compression, dim);
    test_read_write::<i64, _>(n, compression, dim);
    test_read_write::<f32, _>(n, compression, dim);
    test_read_write::<f64, _>(n, compression, dim);
}

fn test_n5_filesystem_dim(dim: usize) {
    let dir = tempdir::TempDir::new("rust_n5_integration_tests").unwrap();
    let path_str = dir.path().to_str().unwrap();

    let n = N5Filesystem::open_or_create(path_str)
        .expect("Failed to create N5 filesystem");
    test_all_types(&n, &CompressionType::Raw(compression::raw::RawCompression::default()), dim);
}

#[test]
fn test_n5_filesystem_dims() {
    for dim in 1..=5 {
        test_n5_filesystem_dim(dim);
    }
}

fn test_all_compressions<N5: N5Reader + N5Writer>(n: &N5) {
    test_all_types(n, &CompressionType::Raw(compression::raw::RawCompression::default()), 3);
    #[cfg(feature = "bzip")]
    test_all_types(n, &CompressionType::Bzip2(compression::bzip::Bzip2Compression::default()), 3);
    #[cfg(feature = "gzip")]
    test_all_types(n, &CompressionType::Gzip(compression::gzip::GzipCompression::default()), 3);
    #[cfg(feature = "lz")]
    test_all_types(n, &CompressionType::Lz4(compression::lz::Lz4Compression::default()), 3);
    #[cfg(feature = "xz")]
    test_all_types(n, &CompressionType::Xz(compression::xz::XzCompression::default()), 3);
}

#[test]
fn test_n5_filesystem_compressions() {
    let dir = tempdir::TempDir::new("rust_n5_integration_tests").unwrap();
    let path_str = dir.path().to_str().unwrap();

    let n = N5Filesystem::open_or_create(path_str)
        .expect("Failed to create N5 filesystem");
    test_all_compressions(&n)
}

#[cfg(feature = "use_ndarray")]
#[test]
fn test_read_ndarray() {
    let dir = tempdir::TempDir::new("rust_n5_integration_tests").unwrap();
    let path_str = dir.path().to_str().unwrap();

    let n = N5Filesystem::open_or_create(path_str)
        .expect("Failed to create N5 filesystem");

    let block_size = smallvec![3i32, 4, 2, 1];
    let data_attrs = DatasetAttributes::new(
        smallvec![3, 300, 200, 100],
        block_size.clone(),
        <DataType as TypeReflection<i32>>::get_type_variant(),
        CompressionType::default(),
    );
    let numel = data_attrs.get_block_num_elements();

    let path_name = "test/dataset/group";
    n.create_dataset(path_name, &data_attrs)
        .expect("Failed to create dataset");


    for k in 0..10 {
        let z = block_size[3] * k;
        for j in 0..10 {
            let y = block_size[2] * j;
            for i in 0..10 {
                let x = block_size[1] * i;

                let mut block_data = Vec::<i32>::with_capacity(numel);

                for zo in 0..block_size[3] {
                    for yo in 0..block_size[2] {
                        for xo in 0..block_size[1] {
                            block_data.push(1000 + x + xo);
                            block_data.push(2000 + y + yo);
                            block_data.push(3000 + z + zo);
                        }
                    }
                }

                let block_in = VecDataBlock::new(
                    block_size.clone(),
                    smallvec![0, i64::from(i), i64::from(j), i64::from(k)],
                    block_data);
                n.write_block(path_name, &data_attrs, &block_in)
                    .expect("Failed to write block");
            }
        }
    }

    let bbox = BoundingBox::new(vec![0, 5, 4, 3], vec![3, 35, 15, 7]);
    let a = n.read_ndarray::<i32>("test/dataset/group", &data_attrs, &bbox).unwrap();

    for z in 0..a.shape()[3] {
        for y in 0..a.shape()[2] {
            for x in 0..a.shape()[1] {
                assert_eq!(a[[0, x, y, z]], 1005 + x as i32, "0 {} {} {}: {}", x, y, z, a[[0, x, y, z]]);
                assert_eq!(a[[1, x, y, z]], 2004 + y as i32, "1 {} {} {}: {}", x, y, z, a[[1, x, y, z]]);
                assert_eq!(a[[2, x, y, z]], 3003 + z as i32, "2 {} {} {}: {}", x, y, z, a[[2, x, y, z]]);
            }
        }
    }
}

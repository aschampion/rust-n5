extern crate n5;
extern crate rand;
extern crate tempdir;


use rand::Rng;

use n5::{
    DatasetAttributes,
    DataType,
    N5Reader,
    N5Writer,
    TypeReflection,
    VecDataBlock,
};
use n5::compression::{
    self,
    CompressionType,
};
use n5::filesystem::N5Filesystem;


fn test_read_write<T: 'static + std::fmt::Debug + rand::Rand + Clone + PartialEq, N5: N5Reader + N5Writer>(
        n: &N5,
        compression: &CompressionType,
) where DataType: TypeReflection<T>,
        VecDataBlock<T>: n5::ReadableDataBlock + n5::WriteableDataBlock,
        DataType: n5::DataBlockCreator<std::vec::Vec<T>> {
    let block_size = vec![44i32, 33, 22];
    let data_attrs = DatasetAttributes::new(
        vec![100, 200, 300],
        block_size.clone(),
        <DataType as TypeReflection<T>>::get_type_variant(),
        compression.clone(),
    );
    let numel = block_size.iter().product::<i32>() as usize;
    let mut rng = rand::thread_rng();
    let mut block_data = Vec::<T>::with_capacity(numel);

    for _ in 0..numel {
        block_data.push(rng.gen());
    }

    let block_in = Box::new(VecDataBlock::new(
        block_size,
        vec![0, 0, 0],
        block_data.clone()));

    let path_name = "test/dataset/group";

    n.create_dataset(path_name, &data_attrs)
        .expect("Failed to create dataset");
    n.write_block(path_name, &data_attrs, block_in)
        .expect("Failed to write block");

    let block_out = n.read_block::<T>(path_name, &data_attrs, vec![0, 0, 0])
        .expect("Failed to read block")
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data);

    // TODO: remove
}

fn test_all_types<N5: N5Reader + N5Writer>(
        n: &N5,
        compression: &CompressionType,
) {
    test_read_write::<u8, _>(n, compression);
    test_read_write::<u16, _>(n, compression);
    test_read_write::<u32, _>(n, compression);
    test_read_write::<u64, _>(n, compression);
    test_read_write::<i8, _>(n, compression);
    test_read_write::<i16, _>(n, compression);
    test_read_write::<i32, _>(n, compression);
    test_read_write::<i64, _>(n, compression);
    test_read_write::<f32, _>(n, compression);
    test_read_write::<f64, _>(n, compression);
}

fn test_all_compressions<N5: N5Reader + N5Writer>(n: &N5) {
    test_all_types(n, &CompressionType::Raw(compression::raw::RawCompression::default()));
    #[cfg(feature = "bzip")]
    test_all_types(n, &CompressionType::Bzip2(compression::bzip::Bzip2Compression::default()));
    #[cfg(feature = "gzip")]
    test_all_types(n, &CompressionType::Gzip(compression::gzip::GzipCompression::default()));
    #[cfg(feature = "lz")]
    test_all_types(n, &CompressionType::Lz4(compression::lz::Lz4Compression::default()));
    #[cfg(feature = "xz")]
    test_all_types(n, &CompressionType::Xz(compression::xz::XzCompression::default()));
}

#[test]
fn test_n5_filesystem() {
    let dir = tempdir::TempDir::new("rust_n5_integration_tests").unwrap();
    let path_str = dir.path().to_str().unwrap();

    let n = N5Filesystem::open_or_create(path_str)
        .expect("Failed to create N5 filesystem");
    test_all_compressions(&n)
}

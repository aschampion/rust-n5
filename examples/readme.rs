extern crate n5;

use n5::{
    DatasetAttributes,
    DataBlock,
    DataType,
    N5Reader,
    N5Writer,
    VecDataBlock
};
use n5::compression::CompressionType;
use n5::filesystem::N5Filesystem;

fn n5_roundtrip(root_path: &str) -> std::io::Result<()> {
    let n = N5Filesystem::open_or_create(root_path)?;

    let block_size = vec![44i32, 33, 22];
    let data_attrs = DatasetAttributes::new(
        vec![100, 200, 300],
        block_size.clone(),
        DataType::INT16,
        CompressionType::default(),
    );
    let numel = block_size.iter().product::<i32>() as usize;
    let block_data: Vec<i16> = vec![0i16; numel];

    let block_in = VecDataBlock::new(
        block_size,
        vec![0, 0, 0],
        block_data.clone());

    let path_name = "test/dataset/group";

    n.create_dataset(path_name, &data_attrs)?;
    n.write_block(path_name, &data_attrs, &block_in)?;

    let block_out = n.read_block::<i16>(path_name, &data_attrs, vec![0, 0, 0])?
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data);

    Ok(())
}

fn main() {
    n5_roundtrip("tmp.n5").expect("N5 roundtrip failed!");
}

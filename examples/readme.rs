use n5::prelude::*;
use n5::smallvec::smallvec;

fn n5_roundtrip(root_path: &str) -> std::io::Result<()> {
    let n = N5Filesystem::open_or_create(root_path)?;

    let block_size = smallvec![44i32, 33, 22];
    let data_attrs = DatasetAttributes::new(
        smallvec![100, 200, 300],
        block_size.clone(),
        DataType::INT16,
        CompressionType::default(),
    );
    let block_data: Vec<i16> = vec![0i16; data_attrs.get_block_num_elements()];

    let block_in = VecDataBlock::new(
        block_size,
        smallvec![0, 0, 0],
        block_data.clone());

    let path_name = "test/dataset/group";

    n.create_dataset(path_name, &data_attrs)?;
    n.write_block(path_name, &data_attrs, &block_in)?;

    let block_out = n.read_block::<i16>(path_name, &data_attrs, smallvec![0, 0, 0])?
        .expect("Block is empty");
    assert_eq!(block_out.get_data(), &block_data);

    Ok(())
}

fn main() {
    n5_roundtrip("tmp.n5").expect("N5 roundtrip failed!");
}

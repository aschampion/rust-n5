#![cfg(feature = "use_ndarray")]

use std::iter::FromIterator;

use ndarray::Array;
use smallvec::smallvec;
use rand::Rng;
use rand::distributions::Standard;

use n5::prelude::*;
use n5::ndarray::prelude::*;


#[test]
fn test_read_ndarray() {

    let dir = tempdir::TempDir::new("rust_n5_ndarray_tests").unwrap();
    let path_str = dir.path().to_str().unwrap();

    let n = N5Filesystem::open_or_create(path_str)
        .expect("Failed to create N5 filesystem");

    let block_size = smallvec![3, 4, 2, 1];
    let data_attrs = DatasetAttributes::new(
        smallvec![3, 300, 200, 100],
        block_size.clone(),
        i32::VARIANT,
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
                            block_data.push(1000 + x as i32 + xo as i32);
                            block_data.push(2000 + y as i32 + yo as i32);
                            block_data.push(3000 + z as i32 + zo as i32);
                        }
                    }
                }

                let block_in = VecDataBlock::new(block_data);
                let block_coord = smallvec![0, u64::from(i), u64::from(j), u64::from(k)];
                n.write_block(path_name, &data_attrs, &block_coord, &block_in)
                    .expect("Failed to write block");
            }
        }
    }

    let bbox = BoundingBox::new(smallvec![0, 5, 4, 3], smallvec![3, 35, 15, 7]);
    let a = n.read_ndarray::<i32>(path_name, &data_attrs, &bbox).unwrap();

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




#[test]
fn test_write_read_ndarray() {

    let dir = tempdir::TempDir::new("rust_n5_ndarray_tests").unwrap();
    let path_str = dir.path().to_str().unwrap();

    let n = N5Filesystem::open_or_create(path_str)
        .expect("Failed to create N5 filesystem");

    let block_size = smallvec![3, 4, 2, 1];
    let data_attrs = DatasetAttributes::new(
        smallvec![3, 300, 200, 100],
        block_size.clone(),
        i32::VARIANT,
        CompressionType::default(),
    );

    let path_name = "test/dataset/group";
    n.create_dataset(path_name, &data_attrs)
        .expect("Failed to create dataset");

    let rng = rand::thread_rng();
    let arr_shape = [3, 35, 15, 7];
    let array: Array<i32, _> = Array::from_iter(
            rng.sample_iter(&Standard)
            .take(arr_shape.iter().product()))
        .into_shape(arr_shape.clone()).unwrap()
        .into_dyn();
    let offset = smallvec![0, 5, 4, 3];

    n.write_ndarray(path_name, &data_attrs, offset.clone(), &array, 0).unwrap();

    let bbox = BoundingBox::new(offset, arr_shape.iter().map(|s| *s as u64).collect());
    let a = n.read_ndarray::<i32>("test/dataset/group", &data_attrs, &bbox).unwrap();

    assert_eq!(array, a);
}

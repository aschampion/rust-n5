use std::cmp;
use std::io::Error;

use itertools::Itertools;

use crate::{
    CoordVec,
    DataBlock,
    DataBlockCreator,
    DatasetAttributes,
    DataType,
    GridCoord,
    N5Reader,
    VecDataBlock,
};


pub mod prelude {
    pub use super::{
        BoundingBox,
        N5NdarrayReader,
    };
}


/// Specifes the extents of an axis-aligned bounding box.
#[derive(Debug)]
pub struct BoundingBox {
    offset: GridCoord,
    size: GridCoord,
}

impl BoundingBox {
    pub fn new(offset: GridCoord, size: GridCoord) -> BoundingBox {
        BoundingBox {
            offset,
            size,
        }
    }
}

pub trait N5NdarrayReader : N5Reader {
    /// Read an abitrary bounding box from an N5 volume in an ndarray, reading
    /// blocks in serial as necessary.
    ///
    /// Assumes blocks are column-major and returns a column-major ndarray.
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
        let offset = ndarray::arr1(&bbox.offset);
        let size = ndarray::arr1(&bbox.size);
        let arr_end = &offset + &size;
        let norm_block_size = Array::from_iter(data_attrs.block_size.iter().map(|n| i64::from(*n)));

        let coord_iter = floor_coord.zip(ceil_coord)
            .map(|(min, max)| min..max)
            .multi_cartesian_product();

        let arr_size: CoordVec<usize> = bbox.size.iter().map(|n| *n as usize).collect();
        let arr_size_a = Array::from_iter(arr_size.iter().map(|n| *n as i64));
        let mut arr = Array::zeros(arr_size.f());

        for coord in coord_iter {
            let block_opt = self.read_block(path_name, data_attrs, GridCoord::from(&coord[..]))?;

            if let Some(block) = block_opt {
                let block_size = Array::from_iter(block.get_size().iter().map(|n| i64::from(*n)));
                let block_size_usize: CoordVec<usize> = block.get_size().iter().map(|n| *n as usize).collect();
                let coord_a = Array::from_vec(coord);

                let block_start = &coord_a * &norm_block_size;
                let block_end = &block_start + &block_size;
                let block_min = (&offset - &block_start).mapv(|v| cmp::max(v, 0));
                let block_max = block_size.clone() - (&block_end - &arr_end).mapv(|v| cmp::max(v, 0));

                let arr_start = (&block_start - &offset).mapv(|v| cmp::max(v, 0));
                let arr_end = arr_size_a.clone() - (&arr_end - &block_end).mapv(|v| cmp::max(v, 0));

                let arr_slice: CoordVec<ndarray::SliceOrIndex> = arr_start.iter().zip(&arr_end)
                    .map(|(&start, &end)| ndarray::SliceOrIndex::Slice {
                        start: start as isize,
                        end: Some(end as isize),
                        step: 1,
                    }).collect();
                let mut arr_view = arr.slice_mut(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

                let block_slice: CoordVec<ndarray::SliceOrIndex> = block_min.iter().zip(&block_max)
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
}

impl<T: N5Reader> N5NdarrayReader for T {}

impl DatasetAttributes {
    pub fn coord_iter(&self) -> impl Iterator<Item = Vec<i64>> + ExactSizeIterator {
        let coord_ceil = self.get_dimensions().iter()
            .zip(self.get_block_size().iter())
            .map(|(&d, &s)| (d + i64::from(s) - 1) / i64::from(s))
            .collect::<GridCoord>();

        CoordIterator::new(&coord_ceil)
    }
}

/// Iterator wrapper to provide exact size when iterating over coordinate
/// ranges.
struct CoordIterator<T: Iterator<Item = Vec<i64>>> {
    iter: T,
    accumulator: usize,
    total_coords: usize,
}

impl CoordIterator<itertools::MultiProduct<std::ops::Range<i64>>> {
    fn new(ceil: &[i64]) -> Self {
        CoordIterator {
            iter: ceil.iter()
                .map(|&c| 0..c)
                .multi_cartesian_product(),
            accumulator: 0,
            total_coords: ceil.iter().product::<i64>() as usize,
        }
    }
}

impl<T: Iterator<Item = Vec<i64>>> Iterator for CoordIterator<T> {
    type Item = Vec<i64>;

    fn next(&mut self) -> Option<Self::Item> {
        self.accumulator += 1;
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_coords - self.accumulator;
        (remaining, Some(remaining))
    }
}

impl<T: Iterator<Item = Vec<i64>>> ExactSizeIterator for CoordIterator<T> {
}


#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_dataset_attributes_coord_iter() {
        use std::collections::HashSet;

        let data_attrs = DatasetAttributes {
            dimensions: smallvec![1, 4, 5],
            block_size: smallvec![1, 2, 3],
            data_type: DataType::INT16,
            compression: crate::compression::CompressionType::default(),
        };

        let coords: HashSet<Vec<i64>> = data_attrs.coord_iter().collect();
        let expected: HashSet<Vec<i64>> = vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![0, 1, 1],
        ].into_iter().collect();

        assert_eq!(coords, expected);
    }
}

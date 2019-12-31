use std::cmp;
use std::io::{
    Error,
    ErrorKind,
};
use std::ops::{
    Sub,
};

use itertools::Itertools;
use ndarray::{
    Array,
    ArrayView,
    IxDyn,
    ShapeBuilder,
    SliceInfo,
};

use crate::{
    BlockCoord,
    CoordVec,
    DataBlock,
    DatasetAttributes,
    GridCoord,
    N5Reader,
    N5Writer,
    ReadableDataBlock,
    ReflectedType,
    ReinitDataBlock,
    VecDataBlock,
    WriteableDataBlock,
};


pub mod prelude {
    pub use super::{
        BoundingBox,
        N5NdarrayReader,
        N5NdarrayWriter,
    };
}


/// Specifes the extents of an axis-aligned bounding box.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundingBox {
    offset: GridCoord,
    size: GridCoord,
}

impl BoundingBox {
    pub fn new(offset: GridCoord, size: GridCoord) -> BoundingBox {
        assert_eq!(offset.len(), size.len());

        BoundingBox {
            offset,
            size,
        }
    }

    pub fn size_block(&self) -> BlockCoord {
        self.size.iter().map(|n| *n as u32).collect()
    }

    pub fn size_ndarray_shape(&self) -> CoordVec<usize> {
        self.size.iter().map(|n| *n as usize).collect()
    }

    /// ```
    /// # use n5::ndarray::BoundingBox;
    /// # use n5::smallvec::smallvec;
    /// let mut a = BoundingBox::new(smallvec![0, 0], smallvec![5, 8]);
    /// let b = BoundingBox::new(smallvec![3, 3], smallvec![5, 3]);
    /// let c = BoundingBox::new(smallvec![3, 3], smallvec![2, 3]);
    /// a.intersect(&b);
    /// assert_eq!(a, c);
    /// ```
    pub fn intersect(&mut self, other: &BoundingBox) {
        assert_eq!(self.offset.len(), other.offset.len());

        self.size.iter_mut()
            .zip(self.offset.iter_mut())
            .zip(other.size.iter())
            .zip(other.offset.iter())
            .for_each(|(((s, o), os), oo)| {
                let new_o = std::cmp::max(*oo, *o);
                *s = std::cmp::max(0, std::cmp::min(*s + *o, *oo + *os) - new_o);
                *o = new_o;
            });
    }

    /// ```
    /// # use n5::ndarray::BoundingBox;
    /// # use n5::smallvec::smallvec;
    /// let mut a = BoundingBox::new(smallvec![0, 0], smallvec![5, 8]);
    /// let b = BoundingBox::new(smallvec![3, 3], smallvec![5, 3]);
    /// let c = BoundingBox::new(smallvec![0, 0], smallvec![8, 8]);
    /// a.union(&b);
    /// assert_eq!(a, c);
    /// ```
    pub fn union(&mut self, other: &BoundingBox) {
        assert_eq!(self.offset.len(), other.offset.len());

        self.size.iter_mut()
            .zip(self.offset.iter_mut())
            .zip(other.size.iter())
            .zip(other.offset.iter())
            .for_each(|(((s, o), os), oo)| {
                let new_o = std::cmp::min(*oo, *o);
                *s = std::cmp::max(*s + *o, *oo + *os) - new_o;
                *o = new_o;
            });
    }

    pub fn end(&self) -> impl Iterator<Item=u64> + '_ {
        self.offset.iter().zip(self.size.iter()).map(|(o, s)| o + s)
    }

    pub fn to_ndarray_slice(&self) -> CoordVec<ndarray::SliceOrIndex> {
        self.offset.iter().zip(self.end())
            .map(|(&start, end)| ndarray::SliceOrIndex::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            }).collect()
    }
}

impl Sub<&GridCoord> for BoundingBox {
    type Output = Self;

    fn sub(self, other: &GridCoord) -> Self::Output {
        Self {
            offset: self.offset.iter()
                .zip(other.iter())
                .map(|(s, o)| s - o)
                .collect(),
            size: self.size.clone(),
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
        where VecDataBlock<T>: DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
              T: ReflectedType + num_traits::identities::Zero {

        if bbox.offset.len() != data_attrs.get_ndim() {
            return Err(Error::new(ErrorKind::InvalidData, "Wrong number of dimensions"));
        }

        let mut arr = Array::zeros(bbox.size_ndarray_shape().f());
        let mut block_buff_opt: Option<VecDataBlock<T>> = None;

        for coord in data_attrs.bounded_coord_iter(bbox) {

            let grid_pos = GridCoord::from(&coord[..]);
            let is_block = match block_buff_opt {
                None => {
                    block_buff_opt = self.read_block(path_name, data_attrs, grid_pos.clone())?;
                    block_buff_opt.is_some()
                },
                Some(ref mut block_buff) => {
                    self.read_block_into(path_name, data_attrs, grid_pos.clone(), block_buff)?.is_some()
                }
            };

            // TODO: cannot combine this into condition below until `let_chains` stabilizes.
            if !is_block { continue; }

            if let Some(ref block) = block_buff_opt {

                let block_bb = data_attrs.get_block_bounds(&grid_pos);
                let mut read_bb = bbox.clone();
                read_bb.intersect(&block_bb);
                let arr_read_bb = read_bb.clone() - &bbox.offset;
                let block_read_bb = read_bb.clone() - &block_bb.offset;

                let arr_slice = arr_read_bb.to_ndarray_slice();
                let mut arr_view = arr.slice_mut(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

                let block_slice = block_read_bb.to_ndarray_slice();

                // N5 datasets are stored f-order/column-major.
                let block_data = ArrayView::from_shape(block_bb.size_ndarray_shape().f(), block.get_data())
                    .expect("TODO: block ndarray failed");
                let block_view = block_data.slice(SliceInfo::<_, IxDyn>::new(block_slice).unwrap().as_ref());

                arr_view.assign(&block_view);
            }
        }

        Ok(arr)
    }
}

impl<T: N5Reader> N5NdarrayReader for T {}


pub trait N5NdarrayWriter : N5Writer {
    /// Write an abitrary bounding box from an ndarray into an N5 volume,
    /// writing blocks in serial as necessary.
    fn write_ndarray<T>(
        &self,
        path_name: &str,
        data_attrs: &DatasetAttributes,
        offset: GridCoord,
        array: &ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>,
        fill_val: T,
    ) -> Result<(), Error>
        where VecDataBlock<T>: DataBlock<T> + ReadableDataBlock + WriteableDataBlock,
              T: ReflectedType + num_traits::identities::Zero {

        if array.ndim() != data_attrs.get_ndim() {
            return Err(Error::new(ErrorKind::InvalidData, "Wrong number of dimensions"));
        }
        let bbox = BoundingBox {
            offset,
            size: array.shape().iter().map(|n| *n as u64).collect(),
        };

        for coord in data_attrs.bounded_coord_iter(&bbox) {

            let grid_coord = GridCoord::from(&coord[..]);
            let nom_block_bb = data_attrs.get_block_bounds(&grid_coord);
            let mut write_bb = nom_block_bb.clone();
            write_bb.intersect(&bbox);
            let arr_bb = write_bb.clone() - &bbox.offset;

            let arr_slice = arr_bb.to_ndarray_slice();
            let arr_view = array.slice(SliceInfo::<_, IxDyn>::new(arr_slice).unwrap().as_ref());

            if write_bb == nom_block_bb {

                // No need to read whether there is an extant block if it is
                // going to be entirely overwrriten.
                let block_vec = arr_view.t().iter().cloned().collect();
                let block = VecDataBlock::new(block_vec);

                self.write_block(path_name, data_attrs, &coord.into(), &block)?;

            } else {

                let block_opt = self.read_block(path_name, data_attrs, grid_coord.clone())?;
                let block_bb = data_attrs.get_block_bounds(&grid_coord);
                let mut block_array = match block_opt {
                    Some(block) => {
                        let block_array = Array::from_shape_vec(block_bb.size_ndarray_shape().f(), block.into_data())
                            .expect("TODO: block ndarray failed");
                        block_array
                    },
                    None => {
                        // If no block exists, need to write from its origin.
                        let block_size_usize = block_bb.size_ndarray_shape();

                        let block_array = Array::from_elem(&block_size_usize[..], fill_val.clone()).into_dyn();
                        block_array
                    }
                };

                let block_write_bb = write_bb.clone() - &block_bb.offset;
                let block_slice = block_write_bb.to_ndarray_slice();
                let mut block_view = block_array.slice_mut(SliceInfo::<_, IxDyn>::new(block_slice).unwrap().as_ref());

                block_view.assign(&arr_view);

                let block_vec = block_array.t().iter().cloned().collect();
                let block = VecDataBlock::new(block_vec);

                self.write_block(path_name, data_attrs, &coord.into(), &block)?;
            }
        }

        Ok(())
    }
}

impl<T: N5Writer> N5NdarrayWriter for T {}


impl DatasetAttributes {
    pub fn coord_iter(&self) -> impl Iterator<Item = Vec<u64>> + ExactSizeIterator {
        let coord_ceil = self.get_dimensions().iter()
            .zip(self.get_block_size().iter())
            .map(|(&d, &s)| (d + u64::from(s) - 1) / u64::from(s))
            .collect::<GridCoord>();

        CoordIterator::new(&coord_ceil)
    }

    pub fn bounded_coord_iter(&self, bbox: &BoundingBox) -> impl Iterator<Item = Vec<u64>> + ExactSizeIterator {
        let floor_coord: GridCoord = bbox.offset.iter()
            .zip(&self.block_size)
            .map(|(&o, &bs)| o / u64::from(bs))
            .collect();
        let ceil_coord: GridCoord = bbox.offset.iter()
            .zip(&bbox.size)
            .zip(self.block_size.iter().cloned().map(u64::from))
            .map(|((&o, &s), bs)| (o + s + bs - 1) / bs)
            .collect();

        CoordIterator::floor_ceil(&floor_coord, &ceil_coord)
    }

    pub fn get_bounds(&self) -> BoundingBox {
        BoundingBox {
            offset: smallvec![0; self.dimensions.len()],
            size: self.dimensions.clone(),
        }
    }

    pub fn get_block_bounds(&self, coord: &GridCoord) -> BoundingBox {
        let mut size: GridCoord = self.get_block_size().iter().cloned().map(u64::from).collect();
        let offset: GridCoord = coord.iter()
            .zip(size.iter())
            .map(|(c, s)| c * s).collect();
        size.iter_mut()
            .zip(offset.iter())
            .zip(self.get_dimensions().iter())
            .for_each(|((s, o), d)| *s = cmp::min(*s + *o, *d) - *o);
        BoundingBox { offset, size }
    }
}

/// Iterator wrapper to provide exact size when iterating over coordinate
/// ranges.
struct CoordIterator<T: Iterator<Item = Vec<u64>>> {
    iter: T,
    accumulator: usize,
    total_coords: usize,
}

impl CoordIterator<itertools::MultiProduct<std::ops::Range<u64>>> {
    fn new(ceil: &[u64]) -> Self {
        CoordIterator {
            iter: ceil.iter()
                .map(|&c| 0..c)
                .multi_cartesian_product(),
            accumulator: 0,
            total_coords: ceil.iter().product::<u64>() as usize,
        }
    }

    fn floor_ceil(floor: &[u64], ceil: &[u64]) -> Self {
        let total_coords = floor.iter()
                .zip(ceil.iter())
                .map(|(&f, &c)| c - f)
                .product::<u64>() as usize;
        CoordIterator {
            iter: floor.iter()
                .zip(ceil.iter())
                .map(|(&f, &c)| f..c)
                .multi_cartesian_product(),
            accumulator: 0,
            total_coords,
        }
    }
}

impl<T: Iterator<Item = Vec<u64>>> Iterator for CoordIterator<T> {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        self.accumulator += 1;
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_coords - self.accumulator;
        (remaining, Some(remaining))
    }
}

impl<T: Iterator<Item = Vec<u64>>> ExactSizeIterator for CoordIterator<T> {
}


#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::DataType;

    #[test]
    fn test_dataset_attributes_coord_iter() {
        use std::collections::HashSet;

        let data_attrs = DatasetAttributes {
            dimensions: smallvec![1, 4, 5],
            block_size: smallvec![1, 2, 3],
            data_type: DataType::INT16,
            compression: crate::compression::CompressionType::default(),
        };

        let coords: HashSet<Vec<u64>> = data_attrs.coord_iter().collect();
        let expected: HashSet<Vec<u64>> = vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![0, 1, 1],
        ].into_iter().collect();

        assert_eq!(coords, expected);
    }
}

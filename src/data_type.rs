use serde::{
    Deserialize,
    Serialize,
};

use crate::BlockHeader;
use crate::VecDataBlock;


/// Data types representable in N5.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
}

#[macro_export]
macro_rules! data_type_match {
    ($match_expr:ident, $ret:ty, $expr:block) => {
        {
            fn inner<RsType: $crate::ReflectedType>() -> $ret $expr;
            match $match_expr {
                $crate::DataType::UINT8 => inner::<u8>(),
                $crate::DataType::UINT16 => inner::<u16>(),
                $crate::DataType::UINT32 => inner::<u32>(),
                $crate::DataType::UINT64 => inner::<u64>(),
                $crate::DataType::INT8 => inner::<i8>(),
                $crate::DataType::INT16 => inner::<i16>(),
                $crate::DataType::INT32 => inner::<i32>(),
                $crate::DataType::INT64 => inner::<i64>(),
                $crate::DataType::FLOAT32 => inner::<f32>(),
                $crate::DataType::FLOAT64 => inner::<f64>(),
            }
        }
    };
}

impl DataType {
    /// Boilerplate method for reflection of primitive type sizes.
    pub fn size_of(self) -> usize {
        data_type_match!(self, usize, {
                std::mem::size_of::<RsType>()
            }
        )
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Trait implemented by primitive types that are reflected in N5.
///
/// The supertraits are not necessary for this trait, but are used to
/// remove redundant bounds elsewhere when operating generically over
/// data types.
pub trait ReflectedType: Clone + Default {
    const VARIANT: DataType;

    fn create_data_block(
        header: BlockHeader,
    ) -> VecDataBlock<Self> {
        VecDataBlock::<Self>::new(
            header.size,
            header.grid_position,
            vec![Self::default(); header.num_el],
        )
    }
}

macro_rules! reflected_type {
    ($d_name:ident, $d_type:ty) => {
        impl ReflectedType for $d_type {
            const VARIANT: DataType = DataType::$d_name;
        }
    }
}

reflected_type!(UINT8,  u8);
reflected_type!(UINT16, u16);
reflected_type!(UINT32, u32);
reflected_type!(UINT64, u64);
reflected_type!(INT8,  i8);
reflected_type!(INT16, i16);
reflected_type!(INT32, i32);
reflected_type!(INT64, i64);
reflected_type!(FLOAT32, f32);
reflected_type!(FLOAT64, f64);

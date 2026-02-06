// GGUF v3 file format parser
// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor, Read};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// Supported tensor data types in GGUF files.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q6_K = 14,
}

impl GgufDType {
    fn from_u32(v: u32) -> io::Result<Self> {
        match v {
            0 => Ok(GgufDType::F32),
            1 => Ok(GgufDType::F16),
            2 => Ok(GgufDType::Q4_0),
            8 => Ok(GgufDType::Q8_0),
            14 => Ok(GgufDType::Q6_K),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF tensor dtype: {}", v),
            )),
        }
    }

    /// Number of bytes per block and elements per block for quantized types.
    pub fn block_size(self) -> (usize, usize) {
        match self {
            GgufDType::F32 => (4, 1),
            GgufDType::F16 => (2, 1),
            GgufDType::Q4_0 => (18, 32),
            GgufDType::Q8_0 => (34, 32),
            GgufDType::Q6_K => (210, 256), // ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per 256 values
        }
    }
}

/// Metadata value types stored in GGUF files.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            GgufValue::I32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgufDType,
    pub offset: usize, // offset relative to data section start
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes of this tensor's data.
    pub fn byte_size(&self) -> usize {
        let n = self.n_elements();
        let (block_bytes, block_elems) = self.dtype.block_size();
        (n / block_elems) * block_bytes
    }
}

/// Parsed GGUF file with metadata, tensor info, and memory-mapped data.
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    mmap: Mmap,
    data_offset: usize, // byte offset where tensor data begins
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap[..]);

        // Read header
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: 0x{:08X}", magic),
            ));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {}", version),
            ));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        // Read metadata key-value pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let value = read_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Read tensor info entries
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let dtype = GgufDType::from_u32(cursor.read_u32::<LittleEndian>()?)?;
            let offset = cursor.read_u64::<LittleEndian>()? as usize;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                },
            );
        }

        // Data section starts at 32-byte aligned offset after header+metadata+tensor info
        let pos = cursor.position() as usize;
        let data_offset = (pos + 31) & !31; // align to 32 bytes

        Ok(GgufFile {
            metadata,
            tensors,
            mmap,
            data_offset,
        })
    }

    /// Get raw byte slice for a tensor's data from the memory-mapped file.
    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.data_offset + info.offset;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }

    /// Convenience: get metadata value by key.
    pub fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a metadata string value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Get a metadata u32 value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    /// Get a metadata f32 value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }
}

// --- Binary reading helpers ---

fn read_string(cursor: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_value(cursor: &mut Cursor<&[u8]>) -> io::Result<GgufValue> {
    let value_type = cursor.read_u32::<LittleEndian>()?;
    read_value_of_type(cursor, value_type)
}

fn read_value_of_type(cursor: &mut Cursor<&[u8]>, value_type: u32) -> io::Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::U8(cursor.read_u8()?)),
        1 => Ok(GgufValue::I8(cursor.read_i8()?)),
        2 => Ok(GgufValue::U16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::I16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::U32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::I32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::F32(cursor.read_f32::<LittleEndian>()?)),
        7 => {
            let b = cursor.read_u8()?;
            Ok(GgufValue::Bool(b != 0))
        }
        8 => Ok(GgufValue::Str(read_string(cursor)?)),
        9 => {
            // Array: element_type (u32) + count (u64) + elements
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_value_of_type(cursor, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::I64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::F64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF value type: {}", value_type),
        )),
    }
}

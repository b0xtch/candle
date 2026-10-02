//! Storage of the Engram embedding tables: on the accelerator, compressed, or offloaded to
//! host memory / memory-mapped files with only the addressed rows moved to the accelerator.
//!
//! Engram addresses are a pure function of the token ids, so the rows a layer needs are known
//! as soon as the batch is. [`MemoryTable::prefetch`] uses this to gather offloaded rows on a
//! background thread while the preceding layers run (section 2.5 of the paper).
use candle::quantized::{k_quants, GgmlDType, GgmlType, QTensor};
use candle::{DType, Device, Result, Tensor};
use std::sync::Arc;

/// A row-addressable table that lives in host memory or in a memory-mapped file.
///
/// Implementations gather rows on the CPU; [`MemoryTable`] takes care of moving them to the
/// compute device.
pub trait HostRowStore: Send + Sync + std::fmt::Debug {
    fn num_rows(&self) -> usize;
    fn row_dim(&self) -> usize;
    /// Gathers the given rows into a CPU tensor of shape `(ids.len(), row_dim)`.
    fn gather(&self, ids: &[u32]) -> Result<Tensor>;
    /// Number of bytes used by the stored rows.
    fn storage_bytes(&self) -> usize;
    /// A short human-readable description of the storage format.
    fn describe(&self) -> String;
}

fn check_ids(ids: &[u32], num_rows: usize) -> Result<()> {
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= num_rows) {
        candle::bail!("engram: row {id} is out of range for a table of {num_rows} rows")
    }
    Ok(())
}

/// Rows held in a dense CPU tensor of any dtype.
#[derive(Debug, Clone)]
pub struct HostTensorRows {
    table: Tensor,
}

impl HostTensorRows {
    pub fn new(table: Tensor) -> Result<Self> {
        table.dims2()?;
        let table = table.to_device(&Device::Cpu)?.contiguous()?;
        Ok(Self { table })
    }
}

impl HostRowStore for HostTensorRows {
    fn num_rows(&self) -> usize {
        self.table.dim(0).unwrap_or(0)
    }

    fn row_dim(&self) -> usize {
        self.table.dim(1).unwrap_or(0)
    }

    fn gather(&self, ids: &[u32]) -> Result<Tensor> {
        check_ids(ids, self.num_rows())?;
        let ids = Tensor::new(ids, &Device::Cpu)?;
        self.table.index_select(&ids, 0)
    }

    fn storage_bytes(&self) -> usize {
        self.table.elem_count() * self.table.dtype().size_in_bytes()
    }

    fn describe(&self) -> String {
        format!("host {:?}", self.table.dtype())
    }
}

/// Rows held in a block-quantized CPU tensor; only the gathered rows are dequantized.
#[derive(Debug, Clone)]
pub struct HostQuantizedRows {
    table: Arc<QTensor>,
    num_rows: usize,
    row_dim: usize,
}

impl HostQuantizedRows {
    pub fn new(table: Arc<QTensor>) -> Result<Self> {
        if !table.device().is_cpu() {
            candle::bail!("engram: HostQuantizedRows expects a CPU QTensor")
        }
        let (num_rows, row_dim) = table.shape().dims2()?;
        Ok(Self {
            table,
            num_rows,
            row_dim,
        })
    }
}

impl HostRowStore for HostQuantizedRows {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn row_dim(&self) -> usize {
        self.row_dim
    }

    fn gather(&self, ids: &[u32]) -> Result<Tensor> {
        check_ids(ids, self.num_rows)?;
        let ids = Tensor::new(ids, &Device::Cpu)?;
        self.table.embedding(&ids)
    }

    fn storage_bytes(&self) -> usize {
        self.table.storage_size_in_bytes()
    }

    fn describe(&self) -> String {
        format!("host {:?}", self.table.dtype())
    }
}

/// Encoding of the rows of a table, used for the tables kept off the compute device and to pick
/// how [`MemoryTable::from_tensor`] compresses a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowFormat {
    /// Plain rows of a float dtype, e.g. `F16` or `BF16`, gathered without conversion.
    Dense(DType),
    /// Rows of GGML blocks, e.g. `Q8_0` or `Q4_0`, dequantized to f32 when gathered.
    Ggml(GgmlDType),
    /// OCP MXFP8: E4M3 values sharing one power-of-two (E8M0) scale per block of 32 values, 8.25
    /// bits per value, decoded to f32 when gathered. DeepSeek-V4.1 ships its Engram tables in
    /// this format, as a `(num_rows, row_dim)` F8_E4M3 tensor of values and a
    /// `(num_rows, row_dim / 32)` F8_E8M0 tensor of scales.
    Mxfp8,
}

impl RowFormat {
    /// GGML's float types are dense formats.
    fn normalize(self) -> Self {
        match self {
            Self::Ggml(GgmlDType::F32) => Self::Dense(DType::F32),
            Self::Ggml(GgmlDType::F16) => Self::Dense(DType::F16),
            Self::Ggml(GgmlDType::BF16) => Self::Dense(DType::BF16),
            format => format,
        }
    }

    /// Bytes taken by the values of a row. MXFP8 rows also have `row_dim / 32` bytes of scales.
    fn row_bytes(&self, row_dim: usize) -> Result<usize> {
        match self {
            Self::Dense(dtype) => Ok(row_dim * dtype.size_in_bytes()),
            Self::Ggml(dtype) => {
                if !row_dim.is_multiple_of(dtype.block_size()) {
                    candle::bail!(
                        "engram: row size {row_dim} is not a multiple of the {dtype:?} block size {}",
                        dtype.block_size()
                    )
                }
                Ok(row_dim / dtype.block_size() * dtype.type_size())
            }
            Self::Mxfp8 => {
                check_mxfp8_row_dim(row_dim)?;
                Ok(row_dim)
            }
        }
    }
}

impl From<DType> for RowFormat {
    fn from(dtype: DType) -> Self {
        Self::Dense(dtype)
    }
}

impl From<GgmlDType> for RowFormat {
    fn from(dtype: GgmlDType) -> Self {
        Self::Ggml(dtype).normalize()
    }
}

/// Number of values sharing a scale in MXFP8.
const MX_BLOCK: usize = 32;

fn check_mxfp8_row_dim(row_dim: usize) -> Result<()> {
    if !row_dim.is_multiple_of(MX_BLOCK) {
        candle::bail!("engram: MXFP8 rows need a multiple of {MX_BLOCK} values, got {row_dim}")
    }
    Ok(())
}

/// The value of every E4M3 bit pattern.
fn e4m3_values() -> &'static [f32; 256] {
    static VALUES: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    VALUES.get_or_init(|| std::array::from_fn(|b| float8::F8E4M3::from_bits(b as u8).to_f32()))
}

/// The power of two encoded by an E8M0 scale.
fn e8m0_to_f32(scale: u8) -> f32 {
    match scale {
        // 2^-127 is an f32 subnormal.
        0 => f32::from_bits(1 << 22),
        255 => f32::NAN,
        e => f32::from_bits((e as u32) << 23),
    }
}

/// Decodes MXFP8 rows: `dst.len()` E4M3 `values` with one E8M0 scale per block of 32.
fn decode_mxfp8(values: &[u8], scales: &[u8], dst: &mut [f32]) {
    let table = e4m3_values();
    let blocks = dst.as_chunks_mut::<MX_BLOCK>().0.iter_mut();
    for ((dst, values), &scale) in blocks.zip(values.as_chunks::<MX_BLOCK>().0).zip(scales) {
        let scale = e8m0_to_f32(scale);
        for (d, &v) in dst.iter_mut().zip(values) {
            *d = table[v as usize] * scale;
        }
    }
}

/// Encodes rows to MXFP8. Each block of 32 values gets the smallest power-of-two scale that
/// keeps it within the E4M3 range (±448), so that nothing saturates, and the values are
/// rounded to the nearest E4M3 number.
fn encode_mxfp8(src: &[f32], values: &mut [u8], scales: &mut [u8]) {
    let blocks = src.as_chunks::<MX_BLOCK>().0.iter();
    let outputs = values.as_chunks_mut::<MX_BLOCK>().0.iter_mut().zip(scales);
    for (src, (values, scale)) in blocks.zip(outputs) {
        let amax = src.iter().fold(0f32, |m, x| m.max(x.abs()));
        let exp = if amax == 0. {
            -127
        } else {
            (amax as f64 / 448.).log2().ceil().clamp(-127., 127.) as i32
        };
        *scale = (exp + 127) as u8;
        let inv_scale = 2f64.powi(-exp);
        for (v, &x) in values.iter_mut().zip(src) {
            *v = float8::F8E4M3::from_f64(x as f64 * inv_scale).to_bits();
        }
    }
}

/// Rows encoded as MXFP8 (see [`RowFormat::Mxfp8`]) in host memory; only the gathered rows are
/// decoded.
#[derive(Debug, Clone)]
pub struct Mxfp8Rows {
    values: Vec<u8>,
    scales: Vec<u8>,
    num_rows: usize,
    row_dim: usize,
}

impl Mxfp8Rows {
    /// Wraps encoded rows: `values` holds the E4M3 values of every row, `scales` the E8M0
    /// scales of every block of 32 values.
    pub fn new(values: Vec<u8>, scales: Vec<u8>, row_dim: usize) -> Result<Self> {
        check_mxfp8_row_dim(row_dim)?;
        if row_dim == 0 || !values.len().is_multiple_of(row_dim) {
            candle::bail!(
                "engram: {} MXFP8 values do not make rows of {row_dim}",
                values.len()
            )
        }
        if scales.len() * MX_BLOCK != values.len() {
            candle::bail!(
                "engram: {} MXFP8 values need {} scales, got {}",
                values.len(),
                values.len() / MX_BLOCK,
                scales.len()
            )
        }
        Ok(Self {
            num_rows: values.len() / row_dim,
            values,
            scales,
            row_dim,
        })
    }

    /// Encodes a dense `(num_rows, row_dim)` table, `row_dim` being a multiple of 32.
    pub fn quantize(table: &Tensor) -> Result<Self> {
        let (num_rows, row_dim) = table.dims2()?;
        check_mxfp8_row_dim(row_dim)?;
        let mut values = vec![0u8; num_rows * row_dim];
        let mut scales = vec![0u8; num_rows * row_dim / MX_BLOCK];
        // Converted by chunks of rows, so that the table is never copied in f32 as a whole.
        let chunk_rows = std::cmp::max(1, (1 << 20) / std::cmp::max(row_dim, 1));
        for start in (0..num_rows).step_by(chunk_rows) {
            let len = std::cmp::min(chunk_rows, num_rows - start);
            let rows = table.narrow(0, start, len)?.to_dtype(DType::F32)?;
            let rows = rows.flatten_all()?.to_vec1::<f32>()?;
            let values = &mut values[start * row_dim..(start + len) * row_dim];
            let scales =
                &mut scales[start * row_dim / MX_BLOCK..(start + len) * row_dim / MX_BLOCK];
            encode_mxfp8(&rows, values, scales);
        }
        Ok(Self {
            values,
            scales,
            num_rows,
            row_dim,
        })
    }

    /// The E4M3 values, row-major.
    pub fn values(&self) -> &[u8] {
        &self.values
    }

    /// The E8M0 scales, one per block of 32 values, row-major.
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
}

impl HostRowStore for Mxfp8Rows {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn row_dim(&self) -> usize {
        self.row_dim
    }

    fn gather(&self, ids: &[u32]) -> Result<Tensor> {
        check_ids(ids, self.num_rows)?;
        let (dim, scales_dim) = (self.row_dim, self.row_dim / MX_BLOCK);
        let mut out = vec![0f32; ids.len() * dim];
        for (dst, &id) in out.chunks_exact_mut(dim).zip(ids) {
            let id = id as usize;
            let values = &self.values[id * dim..(id + 1) * dim];
            decode_mxfp8(
                values,
                &self.scales[id * scales_dim..(id + 1) * scales_dim],
                dst,
            );
        }
        Tensor::from_vec(out, (ids.len(), dim), &Device::Cpu)
    }

    fn storage_bytes(&self) -> usize {
        self.values.len() + self.scales.len()
    }

    fn describe(&self) -> String {
        "host MXFP8".to_string()
    }
}

fn dequantize_row_as<T: GgmlType>(src: &[u8], dst: &mut [f32]) -> Result<()> {
    let size = std::mem::size_of::<T>();
    let aligned = (src.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>());
    if !src.len().is_multiple_of(size) || !aligned {
        candle::bail!("engram: misaligned quantized row")
    }
    // Safety: the length and alignment were checked above and GGML block types are plain old
    // data made of integers and half-precision floats, valid for any bit pattern.
    let blocks = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const T, src.len() / size) };
    T::to_float(blocks, dst);
    Ok(())
}

fn dequantize_row(dtype: GgmlDType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    use k_quants::*;
    match dtype {
        GgmlDType::F32 => dequantize_row_as::<f32>(src, dst),
        GgmlDType::F16 => dequantize_row_as::<half::f16>(src, dst),
        GgmlDType::BF16 => dequantize_row_as::<half::bf16>(src, dst),
        GgmlDType::Q4_0 => dequantize_row_as::<BlockQ4_0>(src, dst),
        GgmlDType::Q4_1 => dequantize_row_as::<BlockQ4_1>(src, dst),
        GgmlDType::Q5_0 => dequantize_row_as::<BlockQ5_0>(src, dst),
        GgmlDType::Q5_1 => dequantize_row_as::<BlockQ5_1>(src, dst),
        GgmlDType::Q8_0 => dequantize_row_as::<BlockQ8_0>(src, dst),
        GgmlDType::Q8_1 => dequantize_row_as::<BlockQ8_1>(src, dst),
        GgmlDType::Q2K => dequantize_row_as::<BlockQ2K>(src, dst),
        GgmlDType::Q3K => dequantize_row_as::<BlockQ3K>(src, dst),
        GgmlDType::Q4K => dequantize_row_as::<BlockQ4K>(src, dst),
        GgmlDType::Q5K => dequantize_row_as::<BlockQ5K>(src, dst),
        GgmlDType::Q6K => dequantize_row_as::<BlockQ6K>(src, dst),
        GgmlDType::Q8K => dequantize_row_as::<BlockQ8K>(src, dst),
    }
}

/// A table read straight from a memory-mapped safetensors or GGUF file.
///
/// Nothing is loaded up front: the operating system pages rows in on first access and keeps
/// the frequently used ones in its page cache. Since n-gram accesses are Zipfian, this gives
/// the paper's multi-level hierarchy for free: hot rows stay in DRAM while the long tail stays
/// on disk (e.g. NVMe), and tables larger than host memory still work.
#[derive(Debug, Clone)]
pub struct MmapRows {
    mmap: Arc<memmap2::Mmap>,
    offset: usize,
    /// Offset of the scales of MXFP8 rows.
    scale_offset: usize,
    num_rows: usize,
    row_dim: usize,
    row_bytes: usize,
    format: RowFormat,
}

impl MmapRows {
    /// Maps a 2D tensor of a safetensors file.
    ///
    /// # Safety
    ///
    /// The file is memory mapped, see [`memmap2::MmapOptions::map`]: it must not be modified
    /// while the table is alive.
    pub unsafe fn from_safetensors<P: AsRef<std::path::Path>>(path: P, name: &str) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = memmap2::MmapOptions::new().map(&file)?;
        let (offset, dtype, shape) = {
            let st = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| candle::Error::Msg(format!("engram: {e}")))?;
            let view = st
                .tensor(name)
                .map_err(|e| candle::Error::Msg(format!("engram: {name}: {e}")))?;
            let offset = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
            let dtype: DType = view.dtype().try_into()?;
            (offset, dtype, view.shape().to_vec())
        };
        let (num_rows, row_dim) = match shape.as_slice() {
            [r, c] => (*r, *c),
            _ => candle::bail!("engram: {name} has shape {shape:?}, expected a 2D table"),
        };
        Self::new(mmap, offset, num_rows, row_dim, RowFormat::Dense(dtype))
    }

    /// Maps an MXFP8 table of a safetensors file, stored as a `(num_rows, row_dim)` F8_E4M3
    /// tensor of values and a `(num_rows, row_dim / 32)` F8_E8M0 (or U8) tensor of scales,
    /// e.g. the `embed.weight` and `embed.scale` tensors of a DeepSeek-V4.1 Engram layer.
    ///
    /// # Safety
    ///
    /// The file is memory mapped, see [`memmap2::MmapOptions::map`]: it must not be modified
    /// while the table is alive.
    pub unsafe fn from_safetensors_mxfp8<P: AsRef<std::path::Path>>(
        path: P,
        name: &str,
        scale_name: &str,
    ) -> Result<Self> {
        use safetensors::Dtype;
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = memmap2::MmapOptions::new().map(&file)?;
        let (offset, scale_offset, shape, scale_shape) = {
            let st = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| candle::Error::Msg(format!("engram: {e}")))?;
            let tensor = |name: &str, dtypes: &[Dtype]| {
                let view = st
                    .tensor(name)
                    .map_err(|e| candle::Error::Msg(format!("engram: {name}: {e}")))?;
                if !dtypes.contains(&view.dtype()) {
                    candle::bail!("engram: {name} is {:?}, expected {dtypes:?}", view.dtype())
                }
                let offset = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
                Ok((offset, view.shape().to_vec()))
            };
            let (offset, shape) = tensor(name, &[Dtype::F8_E4M3])?;
            let (scale_offset, scale_shape) = tensor(scale_name, &[Dtype::F8_E8M0, Dtype::U8])?;
            (offset, scale_offset, shape, scale_shape)
        };
        let (num_rows, row_dim) = match shape.as_slice() {
            [r, c] => (*r, *c),
            _ => candle::bail!("engram: {name} has shape {shape:?}, expected a 2D table"),
        };
        check_mxfp8_row_dim(row_dim)?;
        if scale_shape != [num_rows, row_dim / MX_BLOCK] {
            candle::bail!(
                "engram: {scale_name} has shape {scale_shape:?}, expected [{num_rows}, {}]",
                row_dim / MX_BLOCK
            )
        }
        let mut rows = Self::new(mmap, offset, num_rows, row_dim, RowFormat::Mxfp8)?;
        rows.scale_offset = scale_offset;
        Ok(rows)
    }

    /// Maps a 2D tensor of a GGUF file, which may be quantized (e.g. `q8_0`).
    ///
    /// # Safety
    ///
    /// The file is memory mapped, see [`memmap2::MmapOptions::map`]: it must not be modified
    /// while the table is alive.
    pub unsafe fn from_gguf<P: AsRef<std::path::Path>>(path: P, name: &str) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)?;
        let info = match content.tensor_infos.get(name) {
            Some(info) => info,
            None => candle::bail!("engram: no tensor named {name} in the gguf file"),
        };
        let (num_rows, row_dim) = info.shape.dims2()?;
        let offset = (content.tensor_data_offset + info.offset) as usize;
        let mmap = memmap2::MmapOptions::new().map(&file)?;
        Self::new(mmap, offset, num_rows, row_dim, info.ggml_dtype.into())
    }

    fn new(
        mmap: memmap2::Mmap,
        offset: usize,
        num_rows: usize,
        row_dim: usize,
        format: RowFormat,
    ) -> Result<Self> {
        let row_bytes = format.row_bytes(row_dim)?;
        if offset + num_rows * row_bytes > mmap.len() {
            candle::bail!("engram: the mapped table extends past the end of the file")
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            offset,
            scale_offset: 0,
            num_rows,
            row_dim,
            row_bytes,
            format,
        })
    }

    fn row(&self, id: u32) -> &[u8] {
        let start = self.offset + id as usize * self.row_bytes;
        &self.mmap[start..start + self.row_bytes]
    }

    fn row_scales(&self, id: u32) -> &[u8] {
        let len = self.row_dim / MX_BLOCK;
        let start = self.scale_offset + id as usize * len;
        &self.mmap[start..start + len]
    }
}

impl HostRowStore for MmapRows {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn row_dim(&self) -> usize {
        self.row_dim
    }

    fn gather(&self, ids: &[u32]) -> Result<Tensor> {
        check_ids(ids, self.num_rows)?;
        match self.format {
            RowFormat::Dense(dtype) => {
                let mut bytes = Vec::with_capacity(ids.len() * self.row_bytes);
                for &id in ids {
                    bytes.extend_from_slice(self.row(id));
                }
                Tensor::from_raw_buffer(&bytes, dtype, &[ids.len(), self.row_dim], &Device::Cpu)
            }
            RowFormat::Ggml(dtype) => {
                let mut out = vec![0f32; ids.len() * self.row_dim];
                for (dst, &id) in out.chunks_exact_mut(self.row_dim).zip(ids) {
                    dequantize_row(dtype, self.row(id), dst)?;
                }
                Tensor::from_vec(out, (ids.len(), self.row_dim), &Device::Cpu)
            }
            RowFormat::Mxfp8 => {
                let mut out = vec![0f32; ids.len() * self.row_dim];
                for (dst, &id) in out.chunks_exact_mut(self.row_dim).zip(ids) {
                    decode_mxfp8(self.row(id), self.row_scales(id), dst);
                }
                Tensor::from_vec(out, (ids.len(), self.row_dim), &Device::Cpu)
            }
        }
    }

    fn storage_bytes(&self) -> usize {
        match self.format {
            RowFormat::Mxfp8 => self.num_rows * (self.row_bytes + self.row_dim / MX_BLOCK),
            _ => self.num_rows * self.row_bytes,
        }
    }

    fn describe(&self) -> String {
        match self.format {
            RowFormat::Dense(dtype) => format!("mmap {dtype:?}"),
            RowFormat::Ggml(dtype) => format!("mmap {dtype:?}"),
            RowFormat::Mxfp8 => "mmap MXFP8".to_string(),
        }
    }
}

/// Where a table loaded from a checkpoint is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    /// On the compute device. Lookups are a device-side gather; required for training.
    #[default]
    Device,
    /// In host memory. Lookups gather rows on the CPU and upload only those rows.
    Host,
}

/// How [`MemoryTable::from_tensor`] stores a dense table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableOptions {
    pub placement: Placement,
    /// Re-encode the table, e.g. as `F16`/`BF16`, as a GGML block quantization such as `Q8_0`
    /// or `Q4_0`, or as MXFP8 (the row size must then be a multiple of the block size). `None`
    /// keeps the dtype of the loaded tensor for host tables and the compute dtype for device
    /// tables. MXFP8 tables are decoded on the host, so they need host placement unless the
    /// compute device is the CPU.
    pub compression: Option<RowFormat>,
    /// For host tables, gather rows on a background thread as soon as the token ids are known
    /// instead of when the layer runs.
    pub prefetch: bool,
}

impl Default for TableOptions {
    fn default() -> Self {
        Self {
            placement: Placement::Device,
            compression: None,
            prefetch: true,
        }
    }
}

impl TableOptions {
    /// Host-resident tables gathered ahead of time, optionally compressed.
    pub fn offloaded(compression: Option<RowFormat>) -> Self {
        Self {
            placement: Placement::Host,
            compression,
            prefetch: true,
        }
    }
}

#[derive(Clone, Debug)]
enum Storage {
    Device(Tensor),
    Quantized(Arc<QTensor>),
    Host(Arc<dyn HostRowStore>),
}

/// An embedding table addressed by row ids, wherever it is stored.
#[derive(Clone, Debug)]
pub struct MemoryTable {
    storage: Storage,
    num_rows: usize,
    row_dim: usize,
    device: Device,
    dtype: DType,
    prefetch: bool,
}

impl MemoryTable {
    /// A dense table on the compute device; rows come back in the table's dtype. Lookups are
    /// differentiable, so a `Var`-backed table can be trained.
    pub fn on_device(table: Tensor) -> Result<Self> {
        let (num_rows, row_dim) = table.dims2()?;
        Ok(Self {
            device: table.device().clone(),
            dtype: table.dtype(),
            storage: Storage::Device(table),
            num_rows,
            row_dim,
            prefetch: false,
        })
    }

    /// A block-quantized table, on the compute device or on the CPU. Only the gathered rows are
    /// dequantized; they are returned on `device` in `dtype`.
    pub fn quantized(table: Arc<QTensor>, device: &Device, dtype: DType) -> Result<Self> {
        if table.device().is_cpu() && !device.is_cpu() {
            let store = Arc::new(HostQuantizedRows::new(table)?);
            return Ok(Self::offloaded(store, device, dtype));
        }
        let (num_rows, row_dim) = table.shape().dims2()?;
        Ok(Self {
            storage: Storage::Quantized(table),
            num_rows,
            row_dim,
            device: device.clone(),
            dtype,
            prefetch: false,
        })
    }

    /// A table offloaded to host memory or to a memory-mapped file. Rows are gathered on the
    /// host and only those rows are copied to `device` (as `dtype`).
    pub fn offloaded(store: Arc<dyn HostRowStore>, device: &Device, dtype: DType) -> Self {
        Self {
            num_rows: store.num_rows(),
            row_dim: store.row_dim(),
            storage: Storage::Host(store),
            device: device.clone(),
            dtype,
            prefetch: true,
        }
    }

    /// Stores a dense table (e.g. loaded from a checkpoint) according to `options`, returning
    /// rows on `device` as `dtype`.
    pub fn from_tensor(
        table: Tensor,
        options: &TableOptions,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        table.dims2()?;
        let compression = options.compression.map(RowFormat::normalize);
        let mut table = match (options.placement, compression) {
            (Placement::Device, None) => {
                Self::on_device(table.to_device(device)?.to_dtype(dtype)?)?
            }
            (Placement::Device, Some(RowFormat::Dense(storage_dtype))) => {
                let table = table.to_device(device)?.to_dtype(storage_dtype)?;
                let (num_rows, row_dim) = table.dims2()?;
                Self {
                    storage: Storage::Device(table),
                    num_rows,
                    row_dim,
                    device: device.clone(),
                    dtype,
                    prefetch: false,
                }
            }
            (Placement::Device, Some(RowFormat::Ggml(q))) => {
                let q = QTensor::quantize_onto(&table.to_device(&Device::Cpu)?, q, device)?;
                Self::quantized(Arc::new(q), device, dtype)?
            }
            (Placement::Device, Some(RowFormat::Mxfp8)) if !device.is_cpu() => {
                candle::bail!("engram: MXFP8 tables are decoded on the host, use Placement::Host")
            }
            (_, compression) => {
                let table = table.to_device(&Device::Cpu)?;
                let store: Arc<dyn HostRowStore> = match compression {
                    None => Arc::new(HostTensorRows::new(table)?),
                    Some(RowFormat::Dense(storage_dtype)) => {
                        Arc::new(HostTensorRows::new(table.to_dtype(storage_dtype)?)?)
                    }
                    Some(RowFormat::Ggml(q)) => Arc::new(HostQuantizedRows::new(Arc::new(
                        QTensor::quantize(&table, q)?,
                    ))?),
                    Some(RowFormat::Mxfp8) => Arc::new(Mxfp8Rows::quantize(&table)?),
                };
                Self::offloaded(store, device, dtype)
            }
        };
        table.prefetch = options.prefetch;
        Ok(table)
    }

    /// Enables or disables background gathering for offloaded tables.
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn row_dim(&self) -> usize {
        self.row_dim
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Whether the table lives off the compute device, i.e. lookups copy rows to the device.
    pub fn is_offloaded(&self) -> bool {
        matches!(self.storage, Storage::Host(_)) && !self.device.is_cpu()
    }

    /// Whether the table is a trainable dense tensor whose lookups are differentiable.
    pub fn is_differentiable(&self) -> bool {
        matches!(&self.storage, Storage::Device(t) if t.dtype() == self.dtype)
    }

    /// Bytes taken by the table on the compute device.
    pub fn device_bytes(&self) -> usize {
        if self.is_offloaded() {
            0
        } else {
            self.storage_bytes()
        }
    }

    /// Bytes taken by the table wherever it is stored.
    pub fn storage_bytes(&self) -> usize {
        match &self.storage {
            Storage::Device(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Storage::Quantized(q) => q.storage_size_in_bytes(),
            Storage::Host(store) => store.storage_bytes(),
        }
    }

    /// A short description of the storage, e.g. `"device BF16"` or `"host Q8_0"`.
    pub fn describe(&self) -> String {
        match &self.storage {
            Storage::Device(t) => format!("device {:?}", t.dtype()),
            Storage::Quantized(q) => format!("device {:?}", q.dtype()),
            Storage::Host(store) => store.describe(),
        }
    }

    /// Moves gathered host rows to the compute device, converting on whichever side moves fewer
    /// bytes.
    fn upload(rows: Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        if rows.dtype().size_in_bytes() > dtype.size_in_bytes() {
            rows.to_dtype(dtype)?.to_device(device)
        } else {
            rows.to_device(device)?.to_dtype(dtype)
        }
    }

    /// Gathers the given rows, returning a `(ids.len(), row_dim)` tensor on the compute device.
    pub fn lookup(&self, ids: &[u32]) -> Result<Tensor> {
        match &self.storage {
            Storage::Device(t) => {
                check_ids(ids, self.num_rows)?;
                let ids = Tensor::new(ids, t.device())?;
                t.index_select(&ids, 0)?.to_dtype(self.dtype)
            }
            Storage::Quantized(q) => {
                check_ids(ids, self.num_rows)?;
                let ids = Tensor::new(ids, &q.device())?;
                Self::upload(q.embedding(&ids)?, &self.device, self.dtype)
            }
            Storage::Host(store) => Self::upload(store.gather(ids)?, &self.device, self.dtype),
        }
    }

    /// Starts gathering the given rows. Offloaded tables do the host-side work on a background
    /// thread (when prefetching is enabled), so that it overlaps with whatever the caller does
    /// before [`PendingRows::wait`].
    pub fn prefetch(&self, ids: Vec<u32>) -> Result<PendingRows> {
        match &self.storage {
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Host(store) if self.prefetch => {
                let store = store.clone();
                let handle = std::thread::Builder::new()
                    .name("engram-prefetch".to_string())
                    .spawn(move || store.gather(&ids))?;
                Ok(PendingRows(Pending::Gathering {
                    handle,
                    device: self.device.clone(),
                    dtype: self.dtype,
                }))
            }
            Storage::Host(_) => Ok(PendingRows(Pending::Deferred {
                table: self.clone(),
                ids,
            })),
            // Device-side gathers are queued on the device right away.
            _ => Ok(PendingRows(Pending::Ready(self.lookup(&ids)?))),
        }
    }
}

#[derive(Debug)]
enum Pending {
    Ready(Tensor),
    Deferred {
        table: MemoryTable,
        ids: Vec<u32>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    Gathering {
        handle: std::thread::JoinHandle<Result<Tensor>>,
        device: Device,
        dtype: DType,
    },
}

/// Rows requested with [`MemoryTable::prefetch`].
#[derive(Debug)]
pub struct PendingRows(Pending);

impl PendingRows {
    /// Waits for the rows, returned as a `(num_ids, row_dim)` tensor on the compute device.
    pub fn wait(self) -> Result<Tensor> {
        match self.0 {
            Pending::Ready(t) => Ok(t),
            Pending::Deferred { table, ids } => table.lookup(&ids),
            #[cfg(not(target_arch = "wasm32"))]
            Pending::Gathering {
                handle,
                device,
                dtype,
            } => {
                let rows = match handle.join() {
                    Ok(rows) => rows?,
                    Err(_) => candle::bail!("engram: the prefetch thread panicked"),
                };
                MemoryTable::upload(rows, &device, dtype)
            }
        }
    }
}

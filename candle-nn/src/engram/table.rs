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

/// Layout of the rows of a memory-mapped table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowFormat {
    /// Plain rows of a float dtype, gathered without conversion.
    Dense(DType),
    /// Rows of GGML blocks, dequantized to f32 when gathered.
    Ggml(GgmlDType),
}

impl RowFormat {
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
        }
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
        Self::new(
            mmap,
            offset,
            num_rows,
            row_dim,
            RowFormat::Ggml(info.ggml_dtype),
        )
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
        }
    }

    fn storage_bytes(&self) -> usize {
        self.num_rows * self.row_bytes
    }

    fn describe(&self) -> String {
        match self.format {
            RowFormat::Dense(dtype) => format!("mmap {dtype:?}"),
            RowFormat::Ggml(dtype) => format!("mmap {dtype:?}"),
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
    /// Re-encode the table, e.g. `F16`, `BF16` or a block quantization such as `Q8_0` or
    /// `Q4_0` (the row size must be a multiple of the block size). `None` keeps the dtype of
    /// the loaded tensor for host tables and the compute dtype for device tables.
    pub compression: Option<GgmlDType>,
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
    pub fn offloaded(compression: Option<GgmlDType>) -> Self {
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
        let float_dtype = |q: GgmlDType| match q {
            GgmlDType::F32 => Some(DType::F32),
            GgmlDType::F16 => Some(DType::F16),
            GgmlDType::BF16 => Some(DType::BF16),
            _ => None,
        };
        let mut table = match (options.placement, options.compression) {
            (Placement::Device, None) => {
                Self::on_device(table.to_device(device)?.to_dtype(dtype)?)?
            }
            (Placement::Device, Some(q)) => match float_dtype(q) {
                Some(storage_dtype) => {
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
                None => {
                    let q = QTensor::quantize_onto(&table.to_device(&Device::Cpu)?, q, device)?;
                    Self::quantized(Arc::new(q), device, dtype)?
                }
            },
            (Placement::Host, compression) => {
                let table = table.to_device(&Device::Cpu)?;
                let store: Arc<dyn HostRowStore> = match compression.map(|q| (q, float_dtype(q))) {
                    None => Arc::new(HostTensorRows::new(table)?),
                    Some((_, Some(storage_dtype))) => {
                        Arc::new(HostTensorRows::new(table.to_dtype(storage_dtype)?)?)
                    }
                    Some((q, None)) => Arc::new(HostQuantizedRows::new(Arc::new(
                        QTensor::quantize(&table, q)?,
                    ))?),
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

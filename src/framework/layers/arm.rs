//! ARMv7-A short-descriptor virtual address translation (VMSAv7 without LPAE).
//!
//! A 32-bit virtual address is resolved through at most two levels. The
//! first-level table has 4096 four-byte entries (16 KiB) and is indexed by the
//! top 12 bits; an entry is a 1 MiB *section*, a 16 MiB *supersection*, or the
//! address of a second-level *coarse* table. A coarse table has 256 four-byte
//! entries (1 KiB) indexed by the next 8 bits, each a 64 KiB *large page* or a
//! 4 KiB *small page*. Linux keeps TTBCR.N at zero, so one table covers the whole
//! address space and `swapper_pg_dir` (or a task's `mm->pgd`) is its root.
//!
//! The long-descriptor format (LPAE) is a different walk and is not handled here.

use std::any::Any;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;

use crate::error::{Result, VolatilityError};
use crate::framework::layers::intel::coalesce_mappings;
use crate::framework::layers::segmented::read_via_mapping;
use crate::framework::layers::{DataLayer, LayerContainer, MappingEntry};

/// log2 of the smallest page.
const PAGE_BITS: u32 = 12;
/// First-level table: 4096 entries of 4 bytes.
const L1_TABLE_SIZE: usize = 0x4000;
/// Second-level (coarse) table: 256 entries of 4 bytes.
const L2_TABLE_SIZE: usize = 0x400;

/// First-level descriptor types, in bits [1:0]; `0b00` is a fault and `0b11` is
/// reserved.
const L1_TABLE: u32 = 0b01;
const L1_SECTION: u32 = 0b10;
/// Bit 18 of a section descriptor selects a supersection.
const L1_SUPERSECTION: u32 = 1 << 18;

/// Second-level descriptor types, in bits [1:0]. Small pages use both `0b10`
/// and `0b11` (bit 0 is XN there).
const L2_FAULT: u32 = 0b00;
const L2_LARGE: u32 = 0b01;

/// A short-descriptor paging layer over a physical base layer.
pub struct ArmLayer {
    name: String,
    base_layer: String,
    /// Physical address of the first-level translation table.
    page_map_offset: u64,
    metadata: HashMap<String, String>,
    /// Translation tables read during translation, keyed by physical address.
    table_cache: Mutex<LruCache<u64, Option<Vec<u8>>>>,
}

impl ArmLayer {
    pub fn new(
        name: impl Into<String>,
        base_layer: impl Into<String>,
        page_map_offset: u64,
    ) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("architecture".to_string(), "ARMv7".to_string());
        metadata.insert("mapped".to_string(), "true".to_string());
        Self {
            name: name.into(),
            base_layer: base_layer.into(),
            page_map_offset,
            metadata,
            table_cache: Mutex::new(LruCache::new(NonZeroUsize::new(1024).unwrap())),
        }
    }

    pub fn page_map_offset(&self) -> u64 {
        self.page_map_offset
    }

    pub fn base_layer_name(&self) -> &str {
        &self.base_layer
    }

    pub fn page_size(&self) -> u64 {
        1 << PAGE_BITS
    }

    /// Read a translation table of `size` bytes at `base`, remembering it.
    fn table(&self, layers: &LayerContainer, base: u64, size: usize) -> Option<Vec<u8>> {
        if let Some(cached) = self.table_cache.lock().unwrap().get(&base) {
            return cached.clone();
        }
        let table = layers.read(&self.base_layer, base, size, false).ok();
        self.table_cache.lock().unwrap().put(base, table.clone());
        table
    }

    fn entry(table: &[u8], index: usize) -> u32 {
        let start = index * 4;
        u32::from_le_bytes(table[start..start + 4].try_into().unwrap())
    }

    /// Translate a virtual address to `(physical address, log2 of the page size)`.
    fn translate(&self, layers: &LayerContainer, offset: u64) -> Result<(u64, u32)> {
        if offset > u32::MAX as u64 {
            return Err(VolatilityError::paged(
                &self.name,
                offset,
                32,
                0,
                "Entry outside virtual address range",
            ));
        }
        let fault = |bits: u32, entry: u32, table: &str| {
            VolatilityError::paged(
                &self.name,
                offset,
                bits,
                entry as u64,
                format!("Page fault at entry {entry:#x} in {table}"),
            )
        };

        let l1 = self
            .table(layers, self.page_map_offset, L1_TABLE_SIZE)
            .ok_or_else(|| fault(32, 0, "first-level table"))?;
        let l1_entry = Self::entry(&l1, (offset >> 20) as usize);
        match l1_entry & 0b11 {
            L1_SECTION if l1_entry & L1_SUPERSECTION != 0 => {
                // 16 MiB; the extended physical address bits above 32 are ignored
                let base = (l1_entry & 0xFF00_0000) as u64;
                Ok((base | (offset & 0x00FF_FFFF), 24))
            }
            L1_SECTION => {
                let base = (l1_entry & 0xFFF0_0000) as u64;
                Ok((base | (offset & 0x000F_FFFF), 20))
            }
            L1_TABLE => {
                let l2_base = (l1_entry & 0xFFFF_FC00) as u64;
                let l2 = self
                    .table(layers, l2_base, L2_TABLE_SIZE)
                    .ok_or_else(|| fault(20, l1_entry, "first-level table"))?;
                let l2_entry = Self::entry(&l2, ((offset >> 12) & 0xFF) as usize);
                match l2_entry & 0b11 {
                    L2_FAULT => Err(fault(12, l2_entry, "second-level table")),
                    L2_LARGE => {
                        let base = (l2_entry & 0xFFFF_0000) as u64;
                        Ok((base | (offset & 0xFFFF), 16))
                    }
                    _ => {
                        let base = (l2_entry & 0xFFFF_F000) as u64;
                        Ok((base | (offset & 0xFFF), 12))
                    }
                }
            }
            // fault, or the reserved type
            _ => Err(fault(20, l1_entry, "first-level table")),
        }
    }

    /// Translate a single address, returning the physical offset and the layer
    /// it lands in.
    pub fn translate_single(&self, layers: &LayerContainer, offset: u64) -> Result<(u64, String)> {
        let (mapped, _) = self.translate(layers, offset)?;
        Ok((mapped, self.base_layer.clone()))
    }

    /// The per-page pieces of a range's mapping, handed over one at a time.
    fn walk_mapping_raw(
        &self,
        layers: &LayerContainer,
        offset: u64,
        length: u64,
        ignore_errors: bool,
        on_entry: &mut dyn FnMut(&MappingEntry),
    ) -> Result<()> {
        if length == 0 {
            match self.translate(layers, offset) {
                Ok((mapped_offset, _)) => on_entry(&MappingEntry {
                    offset,
                    size: 0,
                    mapped_offset,
                    mapped_size: 0,
                    layer: self.base_layer.clone(),
                }),
                Err(error) if !ignore_errors => return Err(error),
                Err(_) => {}
            }
            return Ok(());
        }

        let mut offset = offset;
        let mut length = length;
        while length > 0 {
            match self.translate(layers, offset) {
                Ok((chunk_offset, bits)) => {
                    let page_size = 1u64 << bits;
                    let chunk_size = (page_size - (offset % page_size)).min(length);
                    if layers.is_valid(&self.base_layer, chunk_offset, chunk_size) {
                        on_entry(&MappingEntry {
                            offset,
                            size: chunk_size,
                            mapped_offset: chunk_offset,
                            mapped_size: chunk_size,
                            layer: self.base_layer.clone(),
                        });
                    } else if !ignore_errors {
                        return Err(VolatilityError::invalid_address(
                            &self.base_layer,
                            chunk_offset,
                            "Mapped address is not valid in the base layer",
                        ));
                    }
                    if chunk_size >= length {
                        break;
                    }
                    length -= chunk_size;
                    offset += chunk_size;
                }
                Err(error) => {
                    if !ignore_errors {
                        return Err(error);
                    }
                    // Jump past the whole unmapped region the faulting level covers.
                    let bits = error.invalid_bits().unwrap_or(PAGE_BITS);
                    let mask = (1u64 << bits) - 1;
                    let advance = mask + 1 - (offset & mask);
                    if advance >= length {
                        break;
                    }
                    length -= advance;
                    offset += advance;
                }
            }
        }
        Ok(())
    }
}

impl DataLayer for ArmLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> &'static str {
        "ArmShortDescriptor"
    }

    fn class_module(&self) -> &'static str {
        "volatility3.framework.layers.arm"
    }

    fn minimum_address(&self) -> u64 {
        0
    }

    fn maximum_address(&self) -> u64 {
        u32::MAX as u64
    }

    fn is_valid(&self, layers: &LayerContainer, offset: u64, length: u64) -> bool {
        match self.mapping(layers, offset, length, false) {
            Ok(entries) => entries
                .iter()
                .all(|entry| layers.is_valid(&entry.layer, entry.mapped_offset, 1)),
            Err(_) => false,
        }
    }

    fn dependencies(&self) -> Vec<String> {
        vec![self.base_layer.clone()]
    }

    fn mapped_regions(&self, layers: &LayerContainer) -> Vec<(u64, u64)> {
        let length = self.maximum_address() - self.minimum_address() + 1;
        match self.mapping(layers, self.minimum_address(), length, true) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| (entry.offset, entry.size))
                .collect(),
            Err(_) => vec![(self.minimum_address(), length)],
        }
    }

    fn mapping(
        &self,
        layers: &LayerContainer,
        offset: u64,
        length: u64,
        ignore_errors: bool,
    ) -> Result<Vec<MappingEntry>> {
        let mut raw = Vec::new();
        self.walk_mapping_raw(layers, offset, length, ignore_errors, &mut |entry| {
            raw.push(entry.clone())
        })?;
        Ok(coalesce_mappings(raw))
    }

    fn walk_mapping(
        &self,
        layers: &LayerContainer,
        offset: u64,
        length: u64,
        ignore_errors: bool,
        on_entry: &mut dyn FnMut(&MappingEntry),
    ) -> Result<()> {
        self.walk_mapping_raw(layers, offset, length, ignore_errors, on_entry)
    }

    fn read(
        &self,
        layers: &LayerContainer,
        offset: u64,
        length: usize,
        pad: bool,
    ) -> Result<Vec<u8>> {
        read_via_mapping(self, layers, offset, length, pad)
    }

    fn write(&self, layers: &LayerContainer, offset: u64, data: &[u8]) -> Result<()> {
        for entry in self.mapping(layers, offset, data.len() as u64, false)? {
            let start = (entry.offset - offset) as usize;
            layers.write(
                &entry.layer,
                entry.mapped_offset,
                &data[start..start + entry.size as usize],
            )?;
        }
        Ok(())
    }

    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    fn page_size(&self) -> Option<u64> {
        Some(self.page_size())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::layers::physical::BufferLayer;
    use std::sync::Arc;

    const L1: u64 = 0x4000;
    const L2: u64 = 0x8000;

    /// A first-level table with one section, one supersection and one coarse
    /// table holding a small page and a large page.
    fn build_image() -> Vec<u8> {
        let mut memory = vec![0u8; 0x200_0000];
        let write = |memory: &mut Vec<u8>, at: u64, value: u32| {
            let at = at as usize;
            memory[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        // VA 0xC000_0000..+1M -> PA 0x0010_0000 (section)
        write(&mut memory, L1 + (0xC00 * 4), 0x0010_0000 | L1_SECTION);
        // VA 0xD000_0000..+16M -> PA 0x0100_0000 (supersection: 16 identical entries)
        for i in 0..16 {
            write(
                &mut memory,
                L1 + ((0xD00 + i) * 4),
                0x0100_0000 | L1_SUPERSECTION | L1_SECTION,
            );
        }
        // VA 0x0040_0000..+1M -> coarse table at L2
        write(&mut memory, L1 + (0x004 * 4), L2 as u32 | L1_TABLE);
        // VA 0x0040_3000 -> PA 0x0020_0000 (small page, XN set)
        write(&mut memory, L2 + (0x003 * 4), 0x0020_0000 | 0b11);
        // VA 0x0041_0000..+64K -> PA 0x0030_0000 (large page: 16 identical entries)
        for i in 0..16 {
            write(&mut memory, L2 + ((0x010 + i) * 4), 0x0030_0000 | L2_LARGE);
        }
        memory
    }

    fn layer() -> (LayerContainer, ArmLayer) {
        let mut memory = build_image();
        memory[0x0010_0010..0x0010_0014].copy_from_slice(&[1, 2, 3, 4]);
        memory[0x0020_0abc..0x0020_0ac0].copy_from_slice(&[5, 6, 7, 8]);
        let layers = LayerContainer::new();
        layers.add(Arc::new(BufferLayer::new("base", memory)));
        (layers, ArmLayer::new("virtual", "base", L1))
    }

    #[test]
    fn translates_every_descriptor_kind() {
        let (layers, arm) = layer();
        assert_eq!(
            arm.translate(&layers, 0xC000_0010).unwrap(),
            (0x0010_0010, 20)
        );
        assert_eq!(
            arm.translate(&layers, 0xD012_3456).unwrap(),
            (0x0112_3456, 24)
        );
        assert_eq!(
            arm.translate(&layers, 0x0040_3abc).unwrap(),
            (0x0020_0abc, 12)
        );
        assert_eq!(
            arm.translate(&layers, 0x0041_8001).unwrap(),
            (0x0030_8001, 16)
        );
        assert_eq!(
            arm.read(&layers, 0xC000_0010, 4, false).unwrap(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            arm.read(&layers, 0x0040_3abc, 4, false).unwrap(),
            vec![5, 6, 7, 8]
        );
    }

    #[test]
    fn unmapped_addresses_fault_with_their_granule() {
        let (layers, arm) = layer();
        let l1_fault = arm.read(&layers, 0x8000_0000, 4, false).unwrap_err();
        assert_eq!(l1_fault.invalid_bits(), Some(20));
        let l2_fault = arm.read(&layers, 0x0040_5000, 4, false).unwrap_err();
        assert_eq!(l2_fault.invalid_bits(), Some(12));
        assert!(arm.read(&layers, 0x1_0000_0000, 4, false).is_err());
    }

    #[test]
    fn mapping_skips_holes_and_merges_pages() {
        let (layers, arm) = layer();
        // the large page is 16 entries that coalesce into one 64K mapping
        let entries = arm.mapping(&layers, 0x0041_0000, 0x1_0000, false).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            (entries[0].mapped_offset, entries[0].size),
            (0x0030_0000, 0x1_0000)
        );
        // over the whole space, only the four mapped regions come back
        let regions = arm.mapped_regions(&layers);
        assert_eq!(
            regions,
            vec![
                (0x0040_3000, 0x1000),
                (0x0041_0000, 0x1_0000),
                (0xC000_0000, 0x10_0000),
                (0xD000_0000, 0x100_0000),
            ]
        );
    }
}

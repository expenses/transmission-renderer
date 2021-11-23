use spirv_std::integer::Integer;

pub trait GetUnchecked<T> {
    unsafe fn get_unchecked(&self, index: usize) -> &T;
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T;
}

impl<T> GetUnchecked<T> for [T] {
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        asm!(
            "%slice_ptr = OpLoad _ {slice_ptr_ptr}",
            "%data_ptr = OpCompositeExtract _ %slice_ptr 0",
            "%val_ptr = OpAccessChain _ %data_ptr {index}",
            "OpReturnValue %val_ptr",
            slice_ptr_ptr = in(reg) &self,
            index = in(reg) index,
            options(noreturn)
        )
    }

    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        asm!(
            "%slice_ptr = OpLoad _ {slice_ptr_ptr}",
            "%data_ptr = OpCompositeExtract _ %slice_ptr 0",
            "%val_ptr = OpAccessChain _ %data_ptr {index}",
            "OpReturnValue %val_ptr",
            slice_ptr_ptr = in(reg) &self,
            index = in(reg) index,
            options(noreturn)
        )
    }
}

impl<T, const N: usize> GetUnchecked<T> for [T; N] {
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        asm!(
            "%val_ptr = OpAccessChain _ {array_ptr} {index}",
            "OpReturnValue %val_ptr",
            array_ptr = in(reg) self,
            index = in(reg) index,
            options(noreturn)
        )
    }

    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        asm!(
            "%val_ptr = OpAccessChain _ {array_ptr} {index}",
            "OpReturnValue %val_ptr",
            array_ptr = in(reg) self,
            index = in(reg) index,
            options(noreturn)
        )
    }
}

pub fn atomic_i_increment(reference: &mut u32) -> u32 {
    unsafe {
        atomic_i_increment_raw::<
            _,
            { spirv_std::memory::Scope::Device as u32 },
            { spirv_std::memory::Semantics::NONE.bits() as u32 },
        >(reference)
    }
}

unsafe fn atomic_i_increment_raw<I: Integer, const SCOPE: u32, const SEMANTICS: u32>(
    ptr: &mut I,
) -> I {
    let mut old = I::default();

    asm! {
        "%u32 = OpTypeInt 32 0",
        "%scope = OpConstant %u32 {scope}",
        "%semantics = OpConstant %u32 {semantics}",
        "%old = OpAtomicIIncrement _ {ptr} %scope %semantics",
        "OpStore {old} %old",
        scope = const SCOPE,
        semantics = const SEMANTICS,
        ptr = in(reg) ptr,
        old = in(reg) &mut old,
    }

    old
}

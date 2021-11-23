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

pub fn atomic_i_add(reference: &mut u32, value: u32) -> u32 {
    unsafe {
        spirv_std::arch::atomic_i_add::<
            _,
            { spirv_std::memory::Scope::Device as u8 },
            { spirv_std::memory::Semantics::NONE.bits() as u8 },
        >(reference, value)
    }
}

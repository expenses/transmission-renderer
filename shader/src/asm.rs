use spirv_std::integer::Integer;

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

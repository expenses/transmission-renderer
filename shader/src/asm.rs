use spirv_std::integer::Integer;
use spirv_std::glam::UVec4;

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

pub fn subgroup_ballot(
    predicate: bool
) -> UVec4 {
    unsafe {
        asm! {
            "%u32 = OpTypeInt 32 0",
            "%uvec4 = OpTypeVector %u32 4",
            "%predicate = OpLoad _ {predicate}",
            "%scope = OpConstant %u32 {scope}",
            "%ballot = OpGroupNonUniformBallot %uvec4 %scope %predicate",
            "OpReturnValue %ballot",
            predicate = in(reg) &predicate,
            scope = const { spirv_std::memory::Scope::Subgroup as u32 },
            options(noreturn)
        }
    }
}

pub fn subgroup_inverse_ballot(
    ballot: UVec4
) -> bool {
    unsafe {
        asm! {
            "%bool = OpTypeBool",
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%ballot = OpLoad _ {ballot}",
            "%predicate = OpGroupNonUniformInverseBallot %bool %scope %ballot",
            "OpReturnValue %predicate",
            ballot = in(reg) &ballot,
            scope = const { spirv_std::memory::Scope::Subgroup as u32 },
            options(noreturn)
        }
    }
}

pub fn subgroup_elect() -> bool {
    unsafe {
        group_elect::<{ spirv_std::memory::Scope::Subgroup as u32 }>()
    }
}

unsafe fn group_elect<const SCOPE: u32>() -> bool {
    unsafe {
        asm! {
            "%bool = OpTypeBool",
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 {scope}",
            "%elect = OpGroupNonUniformElect %bool %scope",
            "OpReturnValue %elect",
            scope = const SCOPE,
            options(noreturn)
        }
    }
}

use spirv_std::glam::UVec4;
use spirv_std::integer::Integer;
use core::arch::asm;

pub fn atomic_i_increment(reference: &mut u32) -> u32 {
    unsafe {
        spirv_std::arch::atomic_i_increment::<
            _,
            { spirv_std::memory::Scope::Device as u32 },
            { spirv_std::memory::Semantics::NONE.bits() as u32 },
        >(reference)
    }
}

pub fn subgroup_ballot(predicate: bool) -> UVec4 {
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

pub fn subgroup_inverse_ballot(ballot: UVec4) -> bool {
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
    unsafe { group_elect::<{ spirv_std::memory::Scope::Subgroup as u32 }>() }
}

unsafe fn group_elect<const SCOPE: u32>() -> bool {
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

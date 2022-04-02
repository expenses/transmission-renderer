use crate::*;

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn fragment_alpha_clip(
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
) {
    let material = index(materials, material_id);

    let texture_sampler = TextureSampler {
        uv,
        textures,
        sampler: *sampler,
    };

    let mut diffuse = material.diffuse_factor;

    if material.textures.diffuse != -1 {
        diffuse *= texture_sampler.sample(material.textures.diffuse as u32)
    }

    if diffuse.w < material.alpha_clipping_cutoff {
        spirv_std::arch::kill();
    }
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn vertex_alpha_clip(
    position: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_uv: &mut Vec2,
    #[spirv(flat)] out_material_id: &mut u32,
) {
    let instance = index(instances, instance_index);
    let similarity = instance.transform.unpack();

    let position = similarity * position;

    *out_uv = uv;
    *out_material_id = instance.material_id;
    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let similarity = index(instances, instance_index).transform.unpack();

    let position = similarity * position;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

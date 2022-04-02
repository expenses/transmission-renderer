use crate::*;

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn instanced(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    #[spirv(flat)] out_material_id: &mut u32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let instance = index(instances, instance_index);
    let similarity = instance.transform.unpack();

    let position = similarity * position;

    *out_position = position;
    *out_normal = similarity.rotation * normal;
    *out_uv = uv;
    *out_material_id = instance.material_id;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn instanced_with_scale(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    #[spirv(flat)] out_material_id: &mut u32,
    out_scale: &mut f32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let instance = index(instances, instance_index);
    let similarity = instance.transform.unpack();

    let position = similarity * position;

    *out_position = position;
    *out_normal = similarity.rotation * normal;
    *out_uv = uv;
    *out_material_id = instance.material_id;
    *out_scale = similarity.scale;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn fullscreen_tri(
    #[spirv(vertex_index)] vert_idx: i32,
    uv: &mut Vec2,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    *uv = Vec2::new(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let pos = 2.0 * *uv - Vec2::ONE;

    *builtin_pos = Vec4::new(pos.x, pos.y, 0.0, 1.0);
}

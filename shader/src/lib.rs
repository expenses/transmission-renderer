#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr, asm),
    register_attr(spirv),
    no_std
)]

mod asm;
mod lighting;
mod tonemapping;

use asm::atomic_i_increment;
use lighting::{
    calculate_normal, evaluate_lights, evaluate_lights_transmission, get_emission,
    get_material_params,
};
use tonemapping::{BakedLottesTonemapperParams, LottesTonemapper};

use glam_pbr::{ibl_volume_refraction, IblVolumeRefractionParams, PerceptualRoughness, View};
use shared_structs::{
    CullingPushConstants, Instance, MaterialInfo, PointLight, PrimitiveInfo, PushConstants,
    Uniforms, Similarity, MAX_LIGHTS_PER_FROXEL, AccelerationStructureDebuggingUniforms
};
use spirv_std::{
    self as _,
    glam::{UVec3, Vec2, Vec3, Vec4, Vec4Swizzles},
    Image, RuntimeArray, Sampler, num_traits::Float,
    arch::IndexUnchecked,
    ray_tracing::AccelerationStructure
};

type Textures = RuntimeArray<Image!(2D, type=f32, sampled)>;

#[spirv(fragment)]
pub fn fragment_transmission(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(flat)] model_scale: f32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 4)] clamp_sampler: &Sampler,
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] point_lights: &[PointLight],
    #[spirv(descriptor_set = 3, binding = 0)] framebuffer: &Image!(2D, type=f32, sampled),
    output: &mut Vec4,
) {
    let material = index(materials, material_id);

    let texture_sampler = TextureSampler {
        uv,
        textures,
        sampler: *sampler,
    };

    let mut diffuse = material.diffuse_factor;

    if material.textures.diffuse != -1 {
        diffuse *= texture_sampler.sample(material.textures.diffuse as u32);
    }

    let mut transmission_factor = material.transmission_factor;

    if material.textures.transmission != -1 {
        transmission_factor *= texture_sampler
            .sample(material.textures.transmission as u32)
            .x;
    }

    let view_vector = Vec3::from(push_constants.view_position) - position;
    let view = View(view_vector.normalize());

    let normal = calculate_normal(normal, &texture_sampler, material, view_vector, uv);

    let material_params = get_material_params(diffuse, material, &texture_sampler);

    let emission = get_emission(material, &texture_sampler);

    #[cfg(target_feature = "RayQueryKHR")]
    let acceleration_structure = unsafe {
        AccelerationStructure::from_u64(push_constants.acceleration_structure_address)
    };

    let (result, mut transmission) = evaluate_lights_transmission(
        material_params,
        view,
        position,
        normal,
        point_lights,
        uniforms,
        #[cfg(target_feature = "RayQueryKHR")]
        &acceleration_structure,
    );

    let mut thickness = material.thickness_factor;

    if material.textures.thickness != -1 {
        thickness *= texture_sampler.sample(material.textures.thickness as u32).y;
    }

    let ggx_lut_sampler = |normal_dot_view: f32, perceptual_roughness: PerceptualRoughness| {
        let uv = Vec2::new(normal_dot_view, perceptual_roughness.0);

        let texture = unsafe { textures.index(uniforms.ggx_lut_texture_index as usize) };
        let sample: Vec4 = texture.sample(*clamp_sampler, uv);

        sample.xy()
    };

    let framebuffer_sampler = |uv, lod| {
        let sample: Vec4 = framebuffer.sample_by_lod(*clamp_sampler, uv, lod);
        sample.truncate()
    };

    transmission += ibl_volume_refraction(
        IblVolumeRefractionParams {
            proj_view_matrix: push_constants.proj_view,
            position,
            material_params,
            framebuffer_size_x: push_constants.framebuffer_size.x,
            normal,
            view,
            thickness,
            model_scale,
            attenuation_distance: material.attenuation_distance,
            attenuation_colour: material.attenuation_colour.into(),
        },
        framebuffer_sampler,
        ggx_lut_sampler,
    );

    let real_transmission = transmission_factor * transmission;

    let diffuse = result.diffuse.lerp(real_transmission, transmission_factor);

    //*output = (normal.0).extend(1.0);

    *output = (diffuse + result.specular + emission).extend(1.0);
}

#[spirv(fragment)]
pub fn fragment(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] point_lights: &[PointLight],
    hdr_framebuffer: &mut Vec4,
    opaque_sampled_framebuffer: &mut Vec4,
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

    let view_vector = Vec3::from(push_constants.view_position) - position;
    let view = View(view_vector.normalize());

    let normal = calculate_normal(normal, &texture_sampler, material, view_vector, uv);

    let material_params = get_material_params(diffuse, material, &texture_sampler);

    let emission = get_emission(material, &texture_sampler);

    #[cfg(target_feature = "RayQueryKHR")]
    let acceleration_structure = unsafe {
        AccelerationStructure::from_u64(push_constants.acceleration_structure_address)
    };

    let result = evaluate_lights(
        material_params,
        view,
        position,
        normal,
        point_lights,
        uniforms,
        #[cfg(target_feature = "RayQueryKHR")]
        &acceleration_structure,
    );

    let output = (result.diffuse + result.specular + emission).extend(1.0);

    //let output = position.extend(1.0);

    *hdr_framebuffer = output;
    *opaque_sampled_framebuffer = output;
}

struct TextureSampler<'a> {
    textures: &'a Textures,
    sampler: Sampler,
    uv: Vec2,
}

impl<'a> TextureSampler<'a> {
    fn sample(&self, texture_id: u32) -> Vec4 {
        let texture = unsafe { self.textures.index(texture_id as usize) };
        texture.sample(self.sampler, self.uv)
    }
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn depth_pre_pass_alpha_clip(
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
pub fn depth_pre_pass_vertex_alpha_clip(
    position: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_uv: &mut Vec2,
    out_material_id: &mut u32,
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
pub fn depth_pre_pass_instanced(
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

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn vertex_instanced(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    out_material_id: &mut u32,
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
pub fn vertex_instanced_with_scale(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    out_material_id: &mut u32,
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

fn index<T, S: IndexUnchecked<T> + ?Sized>(structure: &S, index: u32) -> &T {
    unsafe { structure.index_unchecked(index as usize) }
}

fn index_mut<T, S: IndexUnchecked<T> + ?Sized>(structure: &mut S, index: u32) -> &mut T {
    unsafe { structure.index_unchecked_mut(index as usize) }
}

mod vk {
    pub struct DrawIndexedIndirectCommand {
        pub index_count: u32,
        pub instance_count: u32,
        pub first_index: u32,
        pub vertex_offset: i32,
        pub first_instance: u32,
    }
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(compute(threads(64)))]
pub fn frustum_culling(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] primitives: &[PrimitiveInfo],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] instance_counts: &mut [u32],
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(push_constant)] push_constants: &CullingPushConstants,
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let instance_id = id.x;

    if instance_id as usize >= instances.len() {
        return;
    }

    let instance = index(instances, instance_id);
    let primitive = index(primitives, instance.primitive_id);

    if cull(
        primitive.packed_bounding_sphere,
        instance.transform.unpack(),
        *push_constants,
    ) {
        return;
    }

    let instance_count = index_mut(instance_counts, instance.primitive_id);

    atomic_i_increment(instance_count);
}

fn cull(
    packed_bounding_sphere: Vec4,
    transform: Similarity,
    push_constants: CullingPushConstants,
) -> bool {
    let mut center = packed_bounding_sphere.truncate();
    center = transform * center;
    center = (push_constants.view * center.extend(1.0)).truncate();
    // in the view, +z = back so we flip it.
    center.z = -center.z;

    let mut radius = packed_bounding_sphere.w;
    radius *= transform.scale;

    let mut visible = (center.z + radius > push_constants.z_near) as u32;

    // Check that object does not cross over either of the left/right/top/bottom planes by
    // radius distance (exploits frustum symmetry with the abs()).
    visible &= (center.z * push_constants.frustum_x_xz.y - center.x.abs() * push_constants.frustum_x_xz.x < radius) as u32;
    visible &= (center.z * push_constants.frustum_y_yz.y - center.y.abs() * push_constants.frustum_y_yz.x < radius) as u32;

    visible == 0
}

const NUM_DRAW_BUFFERS: usize = 4;

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(compute(threads(64)))]
pub fn demultiplex_draws(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] primitives: &[PrimitiveInfo],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] instance_counts: &[u32],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] draw_counts: &mut [u32; NUM_DRAW_BUFFERS],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] opaque_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] alpha_clip_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(descriptor_set = 0, binding = 5, storage_buffer)] transmission_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] transmission_alpha_clip_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let draw_id = id.x;

    if draw_id as usize >= instance_counts.len() {
        return;
    }

    let instance_count = *index(instance_counts, draw_id);

    if instance_count == 0 {
        return;
    }

    let primitive = index(primitives, draw_id);

    let draw_count = index_mut(draw_counts, primitive.draw_buffer_index);

    let non_zero_draw_id = atomic_i_increment(draw_count);

    let draw_command = vk::DrawIndexedIndirectCommand {
        instance_count,
        index_count: primitive.index_count,
        first_index: primitive.first_index,
        first_instance: primitive.first_instance,
        vertex_offset: 0,
    };

    match primitive.draw_buffer_index {
        0 => *index_mut(opaque_draws, non_zero_draw_id) = draw_command,
        1 => *index_mut(alpha_clip_draws, non_zero_draw_id) = draw_command,
        2 => *index_mut(transmission_draws, non_zero_draw_id) = draw_command,
        _ => *index_mut(transmission_alpha_clip_draws, non_zero_draw_id) = draw_command,
    };
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(compute(threads(8, 8)))]
pub fn assign_lights_to_froxels(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] point_lights: &[PointLight],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] froxel_light_counts: &mut [u32],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] froxel_light_indices: &mut [u32],
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let froxel_id = id.x;
    let light_id = id.y;

    if froxel_id as usize >= froxel_light_counts.len() {
        return;
    }

    if light_id as usize >= point_lights.len() {
        return;
    }

    let light_offset = atomic_i_increment(index_mut(froxel_light_counts, froxel_id));

    let global_light_index = froxel_id * MAX_LIGHTS_PER_FROXEL + light_offset;

    *index_mut(froxel_light_indices, global_light_index) = light_id;
}

/*
const DEBUG_COLOURS: [Vec3; 13] = [
    const_vec3!([0.0, 0.0, 0.6647]),      // dark blue
    const_vec3!([0.0, 0.0, 0.9647]),      // blue
    const_vec3!([0.0, 0.9255, 0.9255]),   // cyan
    const_vec3!([0.0, 0.5647, 0.0]),      // dark green
    const_vec3!([0.0, 0.7843, 0.0]),      // green
    const_vec3!([1.0, 1.0, 0.0]),         // yellow
    const_vec3!([0.90588, 0.75294, 0.0]), // yellow-orange
    const_vec3!([1.0, 0.5647, 0.0]),      // orange
    const_vec3!([1.0, 0.0, 0.0]),         // bright red
    const_vec3!([0.8392, 0.0, 0.0]),      // red
    const_vec3!([1.0, 0.0, 1.0]),         // magenta
    const_vec3!([0.6, 0.3333, 0.7882]),   // purple
    const_vec3!([1.0, 1.0, 1.0]),         // white
];

fn debug_colour_for_id(id: u32) -> Vec3 {
    DEBUG_COLOURS[(id as usize % DEBUG_COLOURS.len())]
}
*/

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

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn fragment_tonemap(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] texture: &Image!(2D, type=f32, sampled),
    #[spirv(push_constant)] params: &BakedLottesTonemapperParams,
    output: &mut Vec4,
) {
    let sample: Vec4 = texture.sample(*sampler, uv);

    *output = LottesTonemapper
        .tonemap(sample.truncate(), *params)
        .extend(1.0);
}

#[cfg(target_feature = "RayQueryKHR")]
#[spirv(compute(threads(8, 8)))]
pub fn acceleration_structure_debugging(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] output: &Image!(2D, format=rgba16f, sampled=false),
    #[spirv(descriptor_set = 0, binding = 1, uniform)] uniforms: &AccelerationStructureDebuggingUniforms,
) {
    let id_xy = id.truncate();

    let pixel_center = id_xy.as_vec2() + 0.5;

    let texture_coordinates = pixel_center / uniforms.size.as_vec2();

    let render_coordinates = texture_coordinates * 2.0 - 1.0;

    // Transform [0, 0, 0] in view-space into world space
    let origin = (uniforms.view_inverse * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
    // Transform the render coordinates into project-y space
    let target =
        uniforms.proj_inverse * Vec4::new(render_coordinates.x, render_coordinates.y, 1.0, 1.0);
    let local_direction_vector = target.truncate().normalize();
    // Rotate the location direction vector into a global direction vector.
    let direction = (uniforms.view_inverse * local_direction_vector.extend(0.0)).truncate();

    let acceleration_structure = unsafe {
        AccelerationStructure::from_u64(push_constants.acceleration_structure_address)
    };

    use spirv_std::ray_tracing::{AccelerationStructure, RayFlags, RayQuery, CommittedIntersection};

    spirv_std::ray_query!(let mut ray);

    unsafe {
        ray.initialize(
            &acceleration_structure,
            RayFlags::OPAQUE,
            0xff,
            origin,
            0.01,
            direction,
            1000.0
        );

        while ray.proceed() {}

        let colour = match ray.get_committed_intersection_type() {
            CommittedIntersection::None => Vec3::ZERO,
            _ => {
                let barycentrics: Vec2 = ray.get_committed_intersection_barycentrics();
                Vec3::new(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y)
            }
        };

        output.write(id.truncate(), colour);
    };
}

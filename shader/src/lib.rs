#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr, asm, asm_const, asm_experimental_arch),
    register_attr(spirv),
    no_std
)]

mod asm;
mod debugging;
mod deferred;
mod depth_pre_pass;
mod fragment;
mod lighting;
mod noise;
mod tonemapping;
mod vertex;

pub use debugging::*;
pub use deferred::*;
pub use depth_pre_pass::*;
pub use fragment::*;
pub use vertex::*;

use noise::BlueNoiseSampler;

use asm::atomic_i_increment;
use lighting::{
    calculate_normal, evaluate_lights, evaluate_lights_transmission, get_emission,
    get_material_params, LightParams,
};
use tonemapping::{BakedLottesTonemapperParams, LottesTonemapper};

use core::ops::{Add, Mul};
use glam_pbr::{ibl_volume_refraction, IblVolumeRefractionParams, PerceptualRoughness, View};
use shared_structs::{
    AccelerationStructureDebuggingUniforms, AssignLightsPushConstants, ClusterAabb,
    CullingPushConstants, Instance, Light, MaterialInfo, PrimitiveInfo, PushConstants, Similarity,
    Uniforms, WriteClusterDataPushConstants, MAX_LIGHTS_PER_CLUSTER,
};
use spirv_std::{
    self as _,
    arch::IndexUnchecked,
    glam::{const_vec3, Mat4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4, Vec4Swizzles},
    num_traits::Float,
    ray_tracing::AccelerationStructure,
    Image, RuntimeArray, Sampler,
};

type Textures = RuntimeArray<Image!(2D, type=f32, sampled)>;

fn sample_by_lod_0(textures: &Textures, sampler: Sampler, uv: Vec2, texture_id: u32) -> Vec4 {
    TextureSampler {
        textures,
        sampler,
        uv,
    }
    .sample_by_lod_0(texture_id)
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

    fn sample_by_lod_0(&self, texture_id: u32) -> Vec4 {
        let texture = unsafe { self.textures.index(texture_id as usize) };
        texture.sample_by_lod(self.sampler, self.uv, 0.0)
    }
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
    // todo: wait, why?
    center.z = -center.z;

    let mut radius = packed_bounding_sphere.w;
    radius *= transform.scale;

    let mut visible = center.z + radius > push_constants.z_near;

    // Check that object does not cross over either of the left/right/top/bottom planes by
    // radius distance (exploits frustum symmetry with the abs()).
    visible &= center.z * push_constants.frustum_x_xz.y
        - center.x.abs() * push_constants.frustum_x_xz.x
        < radius;
    visible &= center.z * push_constants.frustum_y_yz.y
        - center.y.abs() * push_constants.frustum_y_yz.x
        < radius;

    !visible
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
#[spirv(compute(threads(4, 4, 4)))]
pub fn write_cluster_data(
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] cluster_data: &mut [ClusterAabb],
    #[spirv(push_constant)] push_constants: &WriteClusterDataPushConstants,
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let cluster_id = id.z * uniforms.num_clusters.x * uniforms.num_clusters.y
        + id.y * uniforms.num_clusters.x
        + id.x;

    if cluster_id as usize >= cluster_data.len() {
        return;
    }

    let cluster_xy = id.truncate();

    let screen_space_min = cluster_xy.as_vec2() * uniforms.cluster_size_in_pixels;
    let screen_space_max = (cluster_xy + 1).as_vec2() * uniforms.cluster_size_in_pixels;

    let screen_to_clip = |mut pos: Vec2| {
        pos /= push_constants.screen_dimensions.as_vec2();
        pos = pos * 2.0 - 1.0;
        Vec4::new(pos.x, pos.y, 0.0, 1.0)
    };

    let clip_to_view = |mut pos: Vec4| {
        pos = push_constants.inverse_perspective * pos;

        pos.truncate() / pos.w
    };

    let view_space_min = clip_to_view(screen_to_clip(screen_space_min));
    let view_space_max = clip_to_view(screen_to_clip(screen_space_max));

    let z_near = uniforms.light_clustering_coefficients.slice_to_depth(id.z);
    let z_far = uniforms
        .light_clustering_coefficients
        .slice_to_depth(id.z + 1);

    let eye = Vec3::new(0.0, 0.0, 1.0);
    let min_point_near = line_intersection_to_z_plane(eye, view_space_min, z_near);
    let min_point_far = line_intersection_to_z_plane(eye, view_space_min, z_far);
    let max_point_near = line_intersection_to_z_plane(eye, view_space_max, z_near);
    let max_point_far = line_intersection_to_z_plane(eye, view_space_max, z_far);

    let cluster = ClusterAabb {
        min: min_point_near
            .min(min_point_far)
            .min(max_point_near)
            .min(max_point_far)
            .into(),
        max: min_point_near
            .max(min_point_far)
            .max(max_point_near)
            .max(max_point_far)
            .into(),
    };

    *index_mut(cluster_data, cluster_id) = cluster;
}

// https://github.com/Angelo1211/HybridRenderingEngine/blob/67a03045e7a96df491b3b0f21ef52c453798eafb/assets/shaders/ComputeShaders/clusterShader.comp#L63-L78
fn line_intersection_to_z_plane(a: Vec3, b: Vec3, z_distance: f32) -> Vec3 {
    // Because this is a Z based normal this is fixed
    let normal = Vec3::new(0.0, 0.0, 1.0);

    let a_to_b = b - a;

    // Computing the intersection length for the line and the plane
    let t = (z_distance - normal.dot(a)) / normal.dot(a_to_b);

    // Computing the actual xyz position of the point along the line
    a + t * a_to_b
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(compute(threads(8, 8)))]
pub fn assign_lights_to_clusters(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] lights: &[Light],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] cluster_light_counts: &mut [u32],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] cluster_light_indices: &mut [u32],
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] cluster_data: &[ClusterAabb],
    #[spirv(push_constant)] push_constants: &AssignLightsPushConstants,
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let cluster_id = id.x;
    let light_id = id.y;

    if cluster_id as usize >= cluster_data.len() {
        return;
    }

    if light_id as usize >= lights.len() {
        return;
    }

    let cluster = index(cluster_data, cluster_id);
    let light = index(lights, light_id);

    let light_position = (push_constants.view_matrix * light.position().extend(1.0)).truncate();

    let falloff_distance_sq = light.colour_emission_and_falloff_distance_sq.w;

    if cluster.distance_sq(light_position) > falloff_distance_sq {
        return;
    }

    if light.is_a_spotlight() {
        // Todo: idk if using the inversed quat from the camera is working 100% here.
        let spotlight_direction =
            push_constants.view_rotation * light.spotlight_direction_and_outer_angle.truncate();

        let angle = light.spotlight_direction_and_outer_angle.w;
        let range = light.colour_emission_and_falloff_distance_sq.w;

        if cluster.cull_spotlight(light_position, spotlight_direction, angle, range) {
            return;
        }
    }

    let light_offset = atomic_i_increment(index_mut(cluster_light_counts, cluster_id));

    let global_light_index = cluster_id * MAX_LIGHTS_PER_CLUSTER + light_offset;

    *index_mut(cluster_light_indices, global_light_index) = light_id;
}

const DEBUG_COLOURS: [Vec3; 15] = [
    const_vec3!([0.0, 0.0, 0.0]),         // black
    const_vec3!([0.0, 0.0, 0.1647]),      // darkest blue
    const_vec3!([0.0, 0.0, 0.3647]),      // darker blue
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
                                          //const_vec3!([1.0, 1.0, 1.0]),         // white
];

fn debug_colour_for_id(id: u32) -> Vec3 {
    *index(&DEBUG_COLOURS, id % DEBUG_COLOURS.len() as u32)
}

struct ModelBuffers<'a> {
    indices: &'a [u32],
    uvs: &'a [Vec2],
}

impl<'a> ModelBuffers<'a> {
    fn get_indices_for_primitive(&self, primitive_index: u32) -> UVec3 {
        let index_offset = primitive_index * 3;

        UVec3::new(
            *index(self.indices, index_offset),
            *index(self.indices, index_offset + 1),
            *index(self.indices, index_offset + 2),
        )
    }

    fn interpolate_uv(&self, indices: UVec3, barycentric_coords: Vec3) -> Vec2 {
        let uv_0 = *index(self.uvs, indices.x);
        let uv_1 = *index(self.uvs, indices.y);
        let uv_2 = *index(self.uvs, indices.z);

        interpolate(uv_0, uv_1, uv_2, barycentric_coords)
    }
}

fn barycentric_coords_from_hits(hit_attributes: Vec2) -> Vec3 {
    Vec3::new(
        1.0 - hit_attributes.x - hit_attributes.y,
        hit_attributes.x,
        hit_attributes.y,
    )
}

fn interpolate<T: Mul<f32, Output = T> + Add<T, Output = T>>(
    a: T,
    b: T,
    c: T,
    barycentric_coords: Vec3,
) -> T {
    a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z
}

fn world_position_from_depth(tex_coord: Vec2, ndc_depth: f32, view_proj_inverse: Mat4) -> Vec3 {
    // Take texture coordinate and remap to [-1.0, 1.0] range.
    let screen_pos = tex_coord * 2.0 - 1.0;

    // Create NDC position.
    let ndc_pos = Vec4::new(screen_pos.x, screen_pos.y, ndc_depth, 1.0);

    // Transform back into world position.
    let world_pos = view_proj_inverse * ndc_pos;

    // Undo projection.
    world_pos.truncate() / world_pos.w
}

#[cfg(target_feature = "RayQueryKHR")]
#[spirv(compute(threads(8, 8)))]
pub fn ray_trace_sun_shadow(
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 1, binding = 0)] depth_buffer: &Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] packed_shadow_bitmasks: &mut [u32],
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let frag_coord = id.truncate().as_vec2();
    let tex_coord = frag_coord / push_constants.framebuffer_size.as_vec2();

    let mut blue_noise_sampler = BlueNoiseSampler {
        textures,
        sampler: *sampler,
        uniforms,
        // I spent an actual hour figuring out that I needed to do this offset. Actually kill me.
        frag_coord: frag_coord + 0.5,
        iteration: 0,
    };

    let depth = {
        let sample: Vec4 = depth_buffer.sample_by_lod(*sampler, tex_coord, 0.0);
        sample.x
    };

    let position = world_position_from_depth(tex_coord, depth, uniforms.proj_view_inverse);

    let acceleration_structure =
        unsafe { AccelerationStructure::from_u64(push_constants.acceleration_structure_address) };

    use spirv_std::ray_tracing::RayQuery;
    spirv_std::ray_query!(let mut shadow_ray);

    let factor = lighting::trace_shadow_ray(
        shadow_ray,
        &acceleration_structure,
        position,
        blue_noise_sampler.sample_directional_light(0.05, uniforms.sun_dir.into()),
        10_000.0,
    );

    let ballot = asm::subgroup_ballot(factor);

    if asm::subgroup_elect() {
        let top_bitmask = ballot.x;
        let bottom_bitmask = ballot.y;

        let (top_row_index, bottom_row_index) = indices_for_block(id, push_constants.framebuffer_size);

        *index_mut(packed_shadow_bitmasks, top_row_index) = top_bitmask;
        *index_mut(packed_shadow_bitmasks, bottom_row_index) = bottom_bitmask;
    }
}

fn div_round_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

fn indices_for_block(id: UVec3, framebuffer_size: UVec2) -> (u32, u32) {
    let block_x = id.x / 8;
    let block_y = id.y / 8;
    let num_blocks_x = div_round_up(framebuffer_size.x, 8);

    let top_row_index = (block_y * 2) * num_blocks_x + block_x;
    let bottom_row_index = (block_y * 2 + 1) * num_blocks_x + block_x;

    (top_row_index, bottom_row_index)
}

#[cfg(target_feature = "RayQueryKHR")]
#[spirv(compute(threads(8, 8)))]
pub fn reconstruct_shadow_buffer(
    #[spirv(descriptor_set = 0, binding = 0)] debug_sun_shadow_buffer: &Image!(2D, format=rgba8, sampled=false),
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] packed_shadow_bitmasks: &[u32],
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let (top_row_index, bottom_row_index) = indices_for_block(id, push_constants.framebuffer_size);

    let top_row = *index(packed_shadow_bitmasks, top_row_index);
    let bottom_row = *index(packed_shadow_bitmasks, bottom_row_index);
    let ballot = UVec4::new(top_row, bottom_row, 0, 0);

    let factor = asm::subgroup_inverse_ballot(ballot);

    unsafe {
        debug_sun_shadow_buffer.write(id.truncate(), Vec4::new(factor as u32 as f32, 0.0, 0.0, 1.0));
    }
}

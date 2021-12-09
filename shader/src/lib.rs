#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr, asm, asm_const, asm_experimental_arch),
    register_attr(spirv),
    no_std
)]

mod asm;
mod lighting;
mod tonemapping;
mod noise;

use noise::BlueNoiseSampler;

use asm::atomic_i_increment;
use lighting::{
    calculate_normal, evaluate_lights, evaluate_lights_transmission, get_emission,
    get_material_params, LightParams,
};
use tonemapping::{BakedLottesTonemapperParams, LottesTonemapper};

use glam_pbr::{ibl_volume_refraction, IblVolumeRefractionParams, PerceptualRoughness, View};
use shared_structs::{
    AccelerationStructureDebuggingUniforms, AssignLightsPushConstants, ClusterAabb,
    CullingPushConstants, Instance, Light, MaterialInfo, PrimitiveInfo, PushConstants, Similarity,
    Uniforms, WriteClusterDataPushConstants, MAX_LIGHTS_PER_CLUSTER,
};
use spirv_std::{
    self as _,
    arch::IndexUnchecked,
    glam::{const_vec3, UVec2, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles},
    num_traits::Float,
    ray_tracing::AccelerationStructure,
    Image, RuntimeArray, Sampler,
};
use core::ops::{Add, Mul};

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
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] lights: &[Light],
    #[spirv(descriptor_set = 2, binding = 1, storage_buffer)] cluster_light_counts: &[u32],
    #[spirv(descriptor_set = 2, binding = 2, storage_buffer)] light_indices: &[u32],
    #[spirv(descriptor_set = 3, binding = 0)] framebuffer: &Image!(2D, type=f32, sampled),
    #[spirv(frag_coord)] frag_coord: Vec4,
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

    let cluster = {
        let cluster_xy = (frag_coord.xy() / uniforms.cluster_size_in_pixels).as_uvec2();

        let cluster_z = uniforms
            .light_clustering_coefficients
            .get_depth_slice(frag_coord.z);

        cluster_z * uniforms.num_clusters.x * uniforms.num_clusters.y
            + cluster_xy.y * uniforms.num_clusters.x
            + cluster_xy.x
    };

    #[cfg(target_feature = "RayQueryKHR")]
    let acceleration_structure =
        unsafe { AccelerationStructure::from_u64(push_constants.acceleration_structure_address) };

    let (result, mut transmission) = evaluate_lights_transmission(
        material_params,
        view,
        position,
        normal,
        uniforms,
        LightParams {
            num_lights: *index(cluster_light_counts, cluster),
            light_indices_offset: cluster * MAX_LIGHTS_PER_CLUSTER,
            light_indices,
            lights,
        },
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

    *output = (diffuse + result.specular + emission).extend(1.0);
}

#[cfg(target_feature = "RayQueryKHR")]
#[spirv(fragment)]
pub fn fragment_sun_shadows_only(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(frag_coord)] frag_coord: Vec4,
    output: &mut Vec4,
) {
    let mut blue_noise_sampler = BlueNoiseSampler {
        textures,
        sampler: *sampler,
        uniforms,
        frag_coord: frag_coord.xy(),
        iteration: 0,
    };

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

    *output = Vec4::new(factor, 0.0, 0.0, 1.0);
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
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] lights: &[Light],
    #[spirv(descriptor_set = 2, binding = 1, storage_buffer)] cluster_light_counts: &[u32],
    #[spirv(descriptor_set = 2, binding = 2, storage_buffer)] light_indices: &[u32],
    #[spirv(frag_coord)] frag_coord: Vec4,
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

    let cluster_z = uniforms
        .light_clustering_coefficients
        .get_depth_slice(frag_coord.z);

    let cluster = {
        let cluster_xy = (frag_coord.xy() / uniforms.cluster_size_in_pixels).as_uvec2();

        cluster_z * uniforms.num_clusters.x * uniforms.num_clusters.y
            + cluster_xy.y * uniforms.num_clusters.x
            + cluster_xy.x
    };

    #[cfg(target_feature = "RayQueryKHR")]
    let acceleration_structure =
        unsafe { AccelerationStructure::from_u64(push_constants.acceleration_structure_address) };

    let mut blue_noise_sampler = BlueNoiseSampler {
        textures,
        sampler: *sampler,
        uniforms,
        frag_coord: frag_coord.xy(),
        iteration: 0,
    };

    let num_lights = *index(cluster_light_counts, cluster);

    let result = evaluate_lights(
        material_params,
        view,
        position,
        normal,
        uniforms,
        LightParams {
            num_lights,
            light_indices_offset: cluster * MAX_LIGHTS_PER_CLUSTER,
            light_indices,
            lights,
        },
        #[cfg(target_feature = "RayQueryKHR")]
        &acceleration_structure,
        #[cfg(target_feature = "RayQueryKHR")]
        &mut blue_noise_sampler
    );

    let mut output = (result.diffuse + result.specular + emission).extend(1.0);

    if uniforms.debug_clusters != 0 {
        output = (debug_colour_for_id(num_lights) + (debug_colour_for_id(cluster) - 0.5) * 0.025)
            .extend(1.0);
        //output = (debug_colour_for_id(cluster_z)).extend(1.0);
    }

    *hdr_framebuffer = output;
    *opaque_sampled_framebuffer = output;
}

fn sample_animated_blue_noise(frag_coord: Vec2, iteration: u32, textures: &Textures, sampler: &Sampler, uniforms: &Uniforms) -> Vec2 {
    let offset = UVec2::new(13, 41);
    let texture_size = Vec2::splat(64.0);

    let first_offset = iteration * 2 * offset;
    let second_offset = (iteration * 2 + 1) * offset;

    let first_sample = sample(textures, *sampler, (frag_coord + first_offset.as_vec2()) / texture_size, uniforms.blue_noise_texture_index).x;
    let second_sample = sample(textures, *sampler, (frag_coord + second_offset.as_vec2()) / texture_size, uniforms.blue_noise_texture_index).x;

    animate_blue_noise(Vec2::new(first_sample, second_sample), uniforms.frame_index)
}

fn animate_blue_noise(blue_noise: Vec2, frame_index: u32) -> Vec2 {
    // The fractional part of the golden ratio
    let golden_ratio_fract = 0.618033988749;
    (blue_noise + (frame_index % 32) as f32 * golden_ratio_fract).fract()
}

fn sample(textures: &Textures, sampler: Sampler, uv: Vec2, texture_id: u32) -> Vec4 {
    TextureSampler {
        textures, sampler, uv
    }.sample(texture_id)
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
        let spotlight_direction = push_constants.view_rotation * light.spotlight_direction_and_outer_angle.truncate();

        let angle = light.spotlight_direction_and_outer_angle.w;
        let range = light.colour_emission_and_falloff_distance_sq.w;

        if cluster.cull_spotlight(light_position, spotlight_direction, angle, range) {
            return
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
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 5, storage_buffer)] indices: &[u32],
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] uvs: &[Vec2],
    #[spirv(descriptor_set = 0, binding = 7, storage_buffer)] primitives: &[PrimitiveInfo],
    #[spirv(descriptor_set = 1, binding = 0)] output: &Image!(2D, format=rgba16f, sampled=false),
    #[spirv(descriptor_set = 1, binding = 1, uniform)] uniforms: &AccelerationStructureDebuggingUniforms,
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] instances: &[Instance],
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

    let acceleration_structure =
        unsafe { AccelerationStructure::from_u64(push_constants.acceleration_structure_address) };

    use spirv_std::ray_tracing::{
        AccelerationStructure, CommittedIntersection, RayFlags, RayQuery,
    };

    let model_buffers = ModelBuffers {
        indices, uvs
    };

    spirv_std::ray_query!(let mut ray);

    unsafe {
        ray.initialize(
            &acceleration_structure,
            RayFlags::NONE,
            0xff,
            origin,
            0.01,
            direction,
            1000.0,
        );

        let mut colour = Vec3::ZERO;

        while ray.proceed() {
            let instance_id = ray.get_candidate_intersection_instance_custom_index();
            let instance = index(instances, instance_id);
            let material = index(materials, instance.material_id);
            let primitive_info = index(primitives, instance.primitive_id);

            let triangle_index = ray.get_candidate_intersection_primitive_index();
            let indices = model_buffers.get_indices_for_primitive(triangle_index + primitive_info.first_index / 3);
            let barycentric_coords = barycentric_coords_from_hits(ray.get_candidate_intersection_barycentrics());
            let interpolated_uv = model_buffers.interpolate_uv(indices, barycentric_coords);

            let texture_sampler = TextureSampler {
                uv: interpolated_uv,
                textures,
                sampler: *sampler,
            };

            let mut diffuse = material.diffuse_factor;

            if material.textures.diffuse != -1 {
                diffuse *= texture_sampler.sample_by_lod_0(material.textures.diffuse as u32)
            }

            if diffuse.w >= material.alpha_clipping_cutoff {
                ray.confirm_intersection();

                colour = diffuse.truncate();
            }
        }

        /*
        let colour = match ray.get_committed_intersection_type() {
            CommittedIntersection::None => Vec3::ZERO,
            _ => {
                barycentric_coords_from_hits(ray.get_committed_intersection_barycentrics())
            }
        };
        */

        output.write(id.truncate(), colour);
    };
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn cluster_debugging_vs(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] cluster_data: &[ClusterAabb],
    #[spirv(vertex_index)] vertex_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_colour: &mut Vec3,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let cluster = index(cluster_data, vertex_index / 8);

    let points = [
        Vec3::from(cluster.min),
        Vec3::new(cluster.min.x, cluster.min.y, cluster.max.z),
        Vec3::from(cluster.min),
        Vec3::new(cluster.min.x, cluster.max.y, cluster.min.z),
        Vec3::from(cluster.min),
        Vec3::new(cluster.max.x, cluster.min.y, cluster.min.z),
        Vec3::new(cluster.max.x, cluster.max.y, cluster.min.z),
        Vec3::from(cluster.max),
    ];

    let position = *index(&points, vertex_index % points.len() as u32);

    *builtin_pos = push_constants.proj_view * position.extend(1.0);

    *out_colour = Vec3::X.lerp(Vec3::Y, (vertex_index % 2) as f32);
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn cluster_debugging_fs(
    colour: Vec3,
    hdr_framebuffer: &mut Vec4,
    opaque_sampled_framebuffer: &mut Vec4,
) {
    let output = colour.extend(1.0);
    *hdr_framebuffer = output;
    *opaque_sampled_framebuffer = output;
}

struct ModelBuffers<'a> {
    indices: &'a [u32],
    uvs: &'a [Vec2]
}

impl<'a> ModelBuffers<'a> {
    fn get_indices_for_primitive(&self, primitive_index: u32) -> UVec3 {
        let index_offset = primitive_index * 3;

        UVec3::new(
            *index(self.indices, index_offset),
            *index(self.indices, index_offset + 1),
            *index(self.indices, index_offset + 2)
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

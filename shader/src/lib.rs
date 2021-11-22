#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]
#![feature(asm)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

use glam_pbr::{
    basic_brdf, ibl_volume_refraction, light_direction_and_attenuation, BasicBrdfParams,
    BrdfResult, IblVolumeRefractionParams, IndexOfRefraction, Light, MaterialParams, Normal,
    PerceptualRoughness, View,
};
use shared_structs::{
    CullingPushConstants, DrawCounts, Instance, MaterialInfo, PackedBoundingSphere, PointLight,
    PrimitiveInfo, PushConstants, Similarity, SunUniform,
};
use spirv_std::{
    glam::{Mat3, UVec3, Vec2, Vec3, Vec4},
    num_traits::Float,
    Image, RuntimeArray, Sampler,
};

type Textures = RuntimeArray<Image!(2D, type=f32, sampled)>;

fn compute_cotangent_frame(normal: Vec3, position: Vec3, uv: Vec2) -> Mat3 {
    // get edge vectors of the pixel triangle
    let delta_pos_1 = spirv_std::arch::ddx_vector(position);
    let delta_pos_2 = spirv_std::arch::ddy_vector(position);
    let delta_uv_1 = spirv_std::arch::ddx_vector(uv);
    let delta_uv_2 = spirv_std::arch::ddy_vector(uv);

    // solve the linear system
    let delta_pos_2_perp = delta_pos_2.cross(normal);
    let delta_pos_1_perp = normal.cross(delta_pos_1);
    let t = delta_pos_2_perp * delta_uv_1.x + delta_pos_1_perp * delta_uv_2.x;
    let b = delta_pos_2_perp * delta_uv_1.y + delta_pos_1_perp * delta_uv_2.y;

    // construct a scale-invariant frame
    let invmax = 1.0 / t.length_squared().max(b.length_squared()).sqrt();
    Mat3::from_cols(t * invmax, b * invmax, normal)
}

#[spirv(fragment)]
pub fn fragment_transmission(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(flat)] model_scale: f32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    //#[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] point_lights: &[PointLight],
    #[spirv(descriptor_set = 0, binding = 1)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 2)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 4, uniform)] sun_uniform: &SunUniform,
    #[spirv(descriptor_set = 0, binding = 5)] clamp_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] framebuffer: &Image!(2D, type=f32, sampled),
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

    let (result, mut transmission) = evaluate_lights_transmission(
        material_params,
        view,
        position,
        normal,
        point_lights,
        sun_uniform,
    );

    let mut thickness = material.thickness_factor;

    if material.textures.thickness != -1 {
        thickness *= texture_sampler.sample(material.textures.thickness as u32).y;
    }

    let ggx_lut_sampler = |normal_dot_view: f32, perceptual_roughness: PerceptualRoughness| {
        let uv = Vec2::new(normal_dot_view, perceptual_roughness.0);

        let texture = unsafe { textures.index(push_constants.ggx_lut_texture_index as usize) };
        let sample: Vec4 = texture.sample(*clamp_sampler, uv);

        Vec2::new(sample.x, sample.y)
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
    //#[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] point_lights: &[PointLight],
    #[spirv(descriptor_set = 0, binding = 1)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 2)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 4, uniform)] sun_uniform: &SunUniform,
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

    let result = evaluate_lights(
        material_params,
        view,
        position,
        normal,
        point_lights,
        sun_uniform,
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

fn calculate_normal(
    interpolated_normal: Vec3,
    texture_sampler: &TextureSampler,
    material: &MaterialInfo,
    view_vector: Vec3,
    uv: Vec2,
) -> Normal {
    let mut normal = interpolated_normal.normalize();

    if material.textures.normal_map != -1 {
        let map_normal = texture_sampler
            .sample(material.textures.normal_map as u32)
            .truncate();
        let map_normal = map_normal * 255.0 / 127.0 - 128.0 / 127.0;

        normal = (compute_cotangent_frame(normal, -view_vector, uv) * map_normal).normalize();
    };

    Normal(normal)
}

fn get_material_params(
    diffuse: Vec4,
    material: &MaterialInfo,
    texture_sampler: &TextureSampler,
) -> MaterialParams {
    let mut metallic = material.metallic_factor;
    let mut roughness = material.roughness_factor;

    if material.textures.metallic_roughness != -1 {
        let sample = texture_sampler.sample(material.textures.metallic_roughness as u32);

        // These two are switched!
        metallic *= sample.z;
        roughness *= sample.y;
    }

    MaterialParams {
        diffuse_colour: diffuse.truncate(),
        metallic,
        perceptual_roughness: PerceptualRoughness(roughness),
        index_of_refraction: IndexOfRefraction(material.index_of_refraction),
    }
}

fn get_emission(material: &MaterialInfo, texture_sampler: &TextureSampler) -> Vec3 {
    let mut emission = Vec3::from(material.emissive_factor);

    if material.textures.emissive != -1 {
        emission *= texture_sampler
            .sample(material.textures.emissive as u32)
            .truncate();
    }

    emission
}

fn evaluate_lights_transmission(
    material_params: MaterialParams,
    view: View,
    position: Vec3,
    normal: Normal,
    point_lights: &[PointLight],
    sun_uniform: &SunUniform,
) -> (BrdfResult, Vec3) {
    let mut sum = basic_brdf(BasicBrdfParams {
        light: Light(sun_uniform.dir.into()),
        light_intensity: sun_uniform.intensity,
        normal,
        view,
        material_params,
    });

    let mut transmission = sun_uniform.intensity
        * glam_pbr::transmission_btdf(material_params, normal, view, Light(sun_uniform.dir.into()));

    let num_lights = point_lights.len() as u32;
    let mut i = 0;

    while i < num_lights {
        let light = index(point_lights, i);

        let (direction, attenuation) =
            light_direction_and_attenuation(position, light.position.into());

        let light_colour = light.colour_and_intensity.truncate();
        let intensity = light.colour_and_intensity.w;

        sum = sum
            + basic_brdf(BasicBrdfParams {
                light: Light(direction),
                light_intensity: light_colour * intensity * attenuation,
                normal,
                view,
                material_params,
            });

        transmission += light_colour
            * intensity
            * attenuation
            * glam_pbr::transmission_btdf(material_params, normal, view, Light(direction));

        i += 1;
    }

    (sum, transmission)
}

fn evaluate_lights(
    material_params: MaterialParams,
    view: View,
    position: Vec3,
    normal: Normal,
    point_lights: &[PointLight],
    sun_uniform: &SunUniform,
) -> BrdfResult {
    let mut sum = basic_brdf(BasicBrdfParams {
        light: Light(sun_uniform.dir.into()),
        light_intensity: sun_uniform.intensity,
        normal,
        view,
        material_params,
    });

    let num_lights = point_lights.len() as u32;
    let mut i = 0;

    while i < num_lights {
        let light = index(point_lights, i);

        let (direction, attenuation) =
            light_direction_and_attenuation(position, light.position.into());

        let light_colour = light.colour_and_intensity.truncate();
        let intensity = light.colour_and_intensity.w;

        sum = sum
            + basic_brdf(BasicBrdfParams {
                light: Light(direction),
                light_intensity: light_colour * intensity * attenuation,
                normal,
                view,
                material_params,
            });

        i += 1;
    }

    sum
}

#[spirv(fragment)]
pub fn depth_pre_pass_alpha_clip(
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(descriptor_set = 0, binding = 1)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 2)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] materials: &[MaterialInfo],
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

#[spirv(vertex)]
pub fn depth_pre_pass_vertex_alpha_clip(
    position: Vec3,
    uv: Vec2,
    material: u32,
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_uv: &mut Vec2,
    out_material: &mut u32,
) {
    let similarity = index(instances, instance_index).transform.unpack();

    let position = similarity * position;

    *out_uv = uv;
    *out_material = material;
    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

#[spirv(vertex)]
pub fn depth_pre_pass_instanced(
    position: Vec3,
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let similarity = index(instances, instance_index).transform.unpack();

    let position = similarity * position;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

#[spirv(vertex)]
pub fn vertex_instanced(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    material: u32,
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    out_material: &mut u32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let similarity = index(instances, instance_index).transform.unpack();

    let position = similarity * position;

    *out_position = position;
    *out_normal = similarity.rotation * normal;
    *out_uv = uv;
    *out_material = material;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

#[spirv(vertex)]
pub fn vertex_instanced_with_scale(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    material: u32,
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    out_material: &mut u32,
    out_scale: &mut f32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let similarity = index(instances, instance_index).transform.unpack();

    let position = similarity * position;

    *out_position = position;
    *out_normal = similarity.rotation * normal;
    *out_uv = uv;
    *out_material = material;
    *out_scale = similarity.scale;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

trait GetUnchecked<T> {
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

fn index<T, S: GetUnchecked<T> + ?Sized>(structure: &S, index: u32) -> &T {
    unsafe { GetUnchecked::get_unchecked(structure, index as usize) }
}

fn index_mut<T, S: GetUnchecked<T> + ?Sized>(structure: &mut S, index: u32) -> &mut T {
    unsafe { GetUnchecked::get_unchecked_mut(structure, index as usize) }
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

#[spirv(compute(threads(64)))]
pub fn frustum_culling(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] primitives: &[PrimitiveInfo],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] instance_counts: &mut [u32],
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
        primitive.bounding_sphere,
        instance.transform.unpack(),
        *push_constants,
    ) {
        return;
    }

    atomic_i_add(index_mut(instance_counts, instance.primitive_id), 1);
}

#[spirv(compute(threads(64)))]
pub fn demultiplex_draws(
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] primitives: &[PrimitiveInfo],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] instance_counts: &[u32],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] draw_counts: &mut [u32; DrawCounts::COUNT],
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] opaque_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(descriptor_set = 0, binding = 5, storage_buffer)] alpha_clip_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] transmission_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(descriptor_set = 0, binding = 7, storage_buffer)] transmission_alpha_clip_draws: &mut [vk::DrawIndexedIndirectCommand],
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let draw_id = id.x;

    if draw_id as usize > instance_counts.len() {
        return;
    }

    let instance_count = *index(instance_counts, draw_id);

    if instance_count == 0 {
        return;
    }

    let primitive = index(primitives, draw_id);

    let mut draw_buffers: [_; DrawCounts::COUNT] = [
        opaque_draws,
        alpha_clip_draws,
        transmission_draws,
        transmission_alpha_clip_draws,
    ];

    let draw_buffer = index_mut(&mut draw_buffers, primitive.draw_buffer_index);
    let draw_count = index_mut(draw_counts, primitive.draw_buffer_index);

    let non_zero_draw_id = atomic_i_add(draw_count, 1);

    let draw_command = index_mut(*draw_buffer, non_zero_draw_id);

    *draw_command = vk::DrawIndexedIndirectCommand {
        instance_count,
        index_count: primitive.index_count,
        first_index: primitive.first_index,
        first_instance: primitive.first_instance,
        vertex_offset: 0,
    };
}

fn cull(
    bounding_sphere: PackedBoundingSphere,
    transform: Similarity,
    push_constants: CullingPushConstants,
) -> bool {
    let mut center = bounding_sphere.center_and_radius.truncate();
    center = transform * center;
    center = (push_constants.view * center.extend(1.0)).truncate();

    let mut radius = bounding_sphere.center_and_radius.w;
    radius *= transform.scale;

    center.z + radius > push_constants.z_near
}

fn atomic_i_add(reference: &mut u32, value: u32) -> u32 {
    unsafe {
        spirv_std::arch::atomic_i_add::<
            _,
            { spirv_std::memory::Scope::Device as u8 },
            { spirv_std::memory::Semantics::NONE.bits() as u8 },
        >(reference, value)
    }
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

#[spirv(fragment)]
pub fn fragment_tonemap(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 2)] sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] texture: &Image!(2D, type=f32, sampled),
    #[spirv(push_constant)] params: &BakedLottesTonemapperParams,
    output: &mut Vec4,
) {
    let sample: Vec4 = texture.sample(*sampler, uv);

    *output = LottesTonemapper
        .tonemap(sample.truncate(), *params)
        .extend(1.0);
}

// This is just lifted from
// https://github.com/termhn/colstodian/blob/f2fb0f55d94644dbb753edd5c01da9a08f0e2d3f/src/tonemap.rs#L187-L220
// because rust-gpu support is hard.

struct LottesTonemapper;

impl LottesTonemapper {
    #[inline]
    fn tonemap_inner(x: f32, params: BakedLottesTonemapperParams) -> f32 {
        let z = x.powf(params.a);
        z / (z.powf(params.d) * params.b + params.c)
    }

    fn tonemap(&self, color: Vec3, params: BakedLottesTonemapperParams) -> Vec3 {
        let max = color.max_element();
        let mut ratio = color / max;
        let tonemapped_max = Self::tonemap_inner(max, params);

        ratio = ratio.powf(params.saturation / params.cross_saturation);
        ratio = ratio.lerp(Vec3::ONE, tonemapped_max.powf(params.crosstalk));
        ratio = ratio.powf(params.cross_saturation);

        (ratio * tonemapped_max).min(Vec3::ONE).max(Vec3::ZERO)
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct BakedLottesTonemapperParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    crosstalk: f32,
    saturation: f32,
    cross_saturation: f32,
}

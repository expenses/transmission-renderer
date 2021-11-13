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
    basic_brdf, BasicBrdfParams, Light, MaterialParams, Normal, PerceptualRoughness, View,
};
use shared_structs::{MaterialInfo, PointLight, PushConstants};
use spirv_std::{
    glam::{const_vec3, Mat3, Vec2, Vec3, Vec4},
    num_traits::Float,
    Image, RuntimeArray, Sampler,
};

type Textures = RuntimeArray<Image!(2D, type=f32, sampled)>;

fn sample_texture(textures: &Textures, sampler: &Sampler, index: u32, uv: Vec2) -> Vec4 {
    let texture = unsafe { textures.index(index as usize) };

    texture.sample(*sampler, uv)
}

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
pub fn fragment(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] point_lights: &[PointLight],
    #[spirv(descriptor_set = 0, binding = 1, uniform)] tonemapper_params: &BakedLottesTonemapperParams,
    #[spirv(descriptor_set = 0, binding = 2)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 3)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] materials: &[MaterialInfo],
    output: &mut Vec4,
) {
    let cluster_id = {
        let cluster_xy =
            (Vec2::new(frag_coord.x, frag_coord.y) / push_constants.tile_size_in_pixels).as_uvec2();
        cluster_xy.y * push_constants.num_tiles.x + cluster_xy.x
    };

    let material = &materials[material_id as usize];

    let diffuse = sample_texture(textures, sampler, material.diffuse_texture, uv);
    let metallic_roughness = sample_texture(textures, sampler, material.diffuse_texture, uv);

    let mut normal = normal.normalize();

    if material.normal_map_texture != -1 {
        let map_normal =
            sample_texture(textures, sampler, material.normal_map_texture as u32, uv).truncate();
        let map_normal = map_normal * 2.0 - 1.0;

        normal = (compute_cotangent_frame(normal, position, uv) * map_normal).normalize();
    };

    let view = View((Vec3::from(push_constants.view_position) - position).normalize());
    let normal = Normal(normal);

    let material_params = MaterialParams {
        diffuse_colour: diffuse.truncate(),
        metallic: metallic_roughness.y,
        perceptual_roughness: PerceptualRoughness(metallic_roughness.z),
        perceptual_dielectric_reflectance: Default::default(),
    };

    let mut colour = Vec3::ZERO;

    colour += basic_brdf(BasicBrdfParams {
        light: Light(push_constants.sun_dir.into()),
        light_intensity: push_constants.sun_intensity.into(),
        normal,
        view,
        material_params,
    });

    let num_lights = point_lights.len();
    let mut i = 0;

    while i < num_lights {
        let light = &point_lights[i];

        let vector = Vec3::from(light.position) - position;
        let distance_sq = vector.length_squared();
        let direction = vector / distance_sq.sqrt();

        let attenuation = 1.0 / distance_sq;

        let light_colour = light.colour_and_intensity.truncate();
        let intensity = light.colour_and_intensity.w;

        colour += basic_brdf(BasicBrdfParams {
            light: Light(direction),
            light_intensity: light_colour * intensity * attenuation,
            normal,
            view,
            material_params,
        });

        i += 1;
    }

    *output = LottesTonemapper
        .tonemap(colour, *tonemapper_params)
        .extend(diffuse.w);
}

#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    material: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    out_material: &mut u32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    *out_position = position;
    *out_normal = normal;
    *out_uv = uv;
    *out_material = material;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

#[spirv(vertex)]
pub fn vertex_instanced(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    material: u32,
    translation: Vec3,
    rotation_0: Vec3,
    rotation_1: Vec3,
    rotation_2: Vec3,
    scale: f32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    out_material: &mut u32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let rotation = Mat3::from_cols(rotation_0, rotation_1, rotation_2);

    let position = (rotation * position) * scale + translation;

    *out_position = position;
    *out_normal = rotation * normal;
    *out_uv = uv;
    *out_material = material;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

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

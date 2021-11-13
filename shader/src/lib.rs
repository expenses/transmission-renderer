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
use shared_structs::{PointLight, PushConstants};
use spirv_std::{
    glam::{const_vec3, Vec2, Vec3, Vec4},
    num_traits::Float,
};

#[spirv(fragment)]
pub fn fragment(
    position: Vec3,
    normal: Vec3,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] point_lights: &[PointLight],
    output: &mut Vec4,
) {
    let cluster_id = {
        let cluster_xy =
            (Vec2::new(frag_coord.x, frag_coord.y) / push_constants.tile_size_in_pixels).as_uvec2();
        cluster_xy.y * push_constants.num_tiles.x + cluster_xy.x
    };

    let view = View((Vec3::from(push_constants.view_position) - position).normalize());
    let normal = Normal(normal.normalize());

    let diffuse_colour = Vec3::new(0.2, 0.8, 0.2);

    let material_params = MaterialParams {
        diffuse_colour,
        metallic: 0.0,
        perceptual_roughness: PerceptualRoughness(0.25),
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

        let direction = Vec3::from(light.position) - position;
        let distance_sq = direction.length_squared();

        let attenuation = 1.0 / distance_sq;

        let light_colour = light.colour_and_intensity.truncate();
        let intensity = light.colour_and_intensity.w;

        let direction = direction / distance_sq.sqrt();

        colour += basic_brdf(BasicBrdfParams {
            light: Light(direction),
            light_intensity: light_colour * intensity * attenuation,
            normal,
            view,
            material_params,
        });

        i += 1;
    }

    *output = colour.extend(1.0);
}

#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    normal: Vec3,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    *out_position = position;
    *out_normal = normal;

    *builtin_pos = push_constants.proj_view * position.extend(1.0);
}

const DEBUG_COLOURS: [Vec3; 16] = [
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
    const_vec3!([1.0, 1.0, 1.0]),         // white
];

fn debug_colour_for_id(id: u32) -> Vec3 {
    DEBUG_COLOURS[(id as usize % DEBUG_COLOURS.len())]
}

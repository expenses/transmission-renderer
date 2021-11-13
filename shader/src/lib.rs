#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]
#![feature(asm)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

use glam_pbr::{basic_brdf, BasicBrdfParams, Light, Normal, PerceptualRoughness, View};
use shared_structs::PushConstants;
use spirv_std::glam::{Vec3, Vec4};

#[spirv(fragment)]
pub fn fragment(
    normal: Vec3,
    view: Vec3,
    #[spirv(push_constant)] push_constants: &PushConstants,
    output: &mut Vec4,
) {
    let colour = basic_brdf(BasicBrdfParams {
        light: Light(push_constants.sun_dir.into()),
        light_intensity: push_constants.sun_intensity.into(),
        normal: Normal(normal.normalize()),
        view: View(view.normalize()),
        diffuse_colour: Vec3::new(0.2, 0.8, 0.2),
        metallic: 0.0,
        perceptual_roughness: PerceptualRoughness(0.75),
        perceptual_dielectric_reflectance: Default::default(),
    });

    *output = colour.extend(1.0);
}

#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    normal: Vec3,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_normal: &mut Vec3,
    out_view: &mut Vec3,
) {
    let position = (push_constants.proj_view * position.extend(1.0)).truncate();

    *out_normal = normal;
    *out_view = Vec3::from(push_constants.view_position) - position;
}

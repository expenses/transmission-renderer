use crate::{sample, Textures};
use shared_structs::Uniforms;
use spirv_std::{
    self as _,
    arch::IndexUnchecked,
    glam::{const_vec3, UVec2, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles},
    num_traits::Float,
    ray_tracing::AccelerationStructure,
    Image, RuntimeArray, Sampler,
};

pub struct BlueNoiseSampler<'a> {
    pub textures: &'a Textures,
    pub sampler: Sampler,
    pub uniforms: &'a Uniforms,
    pub frag_coord: Vec2,
    pub iteration: u32,
}

impl<'a> BlueNoiseSampler<'a> {
    // Sample a random vec2 from a blue noise texture.
    //
    // See Ray Tracing Gems II, Chapter 24.7.2, Page 381 & 382.
    pub fn sample(&mut self) -> Vec2 {
        let offset = UVec2::new(13, 41);
        let texture_size = Vec2::splat(64.0);

        let first_offset = self.iteration * offset;
        self.iteration += 1;
        let second_offset = self.iteration * offset;
        self.iteration += 1;

        let first_sample = sample(
            self.textures,
            self.sampler,
            (self.frag_coord + first_offset.as_vec2()) / texture_size,
            self.uniforms.blue_noise_texture_index,
        )
        .x;
        let second_sample = sample(
            self.textures,
            self.sampler,
            (self.frag_coord + second_offset.as_vec2()) / texture_size,
            self.uniforms.blue_noise_texture_index,
        )
        .x;

        animate_blue_noise(
            Vec2::new(first_sample, second_sample),
            self.uniforms.frame_index,
        )
    }

    // Maps 2 randomly generated numbers from 0 to 1 onto a circle with a radius of 1.
    //
    // See Ray Tracing Gems II, Chapter 24.7.2, Page 381.
    pub fn sample_unit_sphere(&mut self) -> Vec2 {
        let rng = self.sample();

        let radius = rng.x.sqrt();
        let angle = rng.y * core::f32::consts::TAU;

        radius * Vec2::new(angle.cos(), angle.sin())
    }

    // Randomly pick a direction vector that points towards a directional light of a given radius.
    //
    // See Ray Tracing Gems II, Chapter 24.7.2, Page 381.
    pub fn sample_directional_light(&mut self, light_radius: f32, center_direction: Vec3) -> Vec3 {
        let point = self.sample_unit_sphere() * light_radius;

        let tangent = center_direction.cross(Vec3::Y).normalize();
        let bitangent = tangent.cross(center_direction).normalize();

        (center_direction + point.x * tangent + point.y * bitangent).normalize()
    }
}

// Animate blue noise over time using the golden ratio.
//
// See Ray Tracing Gems II, Chapter 24.7.2, Page 383 & 384.
fn animate_blue_noise(blue_noise: Vec2, frame_index: u32) -> Vec2 {
    // The fractional part of the golden ratio
    let golden_ratio_fract = 0.618033988749;
    (blue_noise + (frame_index % 32) as f32 * golden_ratio_fract).fract()
}

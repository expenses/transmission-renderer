#![no_std]

use glam::{Mat4, UVec2, Vec2, Vec3, Vec3A, Vec4};

pub struct PushConstants {
    pub proj_view: Mat4,
    pub view_position: Vec3A,
    pub tile_size_in_pixels: Vec2,
    pub num_tiles: UVec2,
    pub debug_froxels: u32,
    //pub depth_range: f32,
}

pub struct SunUniform {
    pub dir: Vec3A,
    pub intensity: Vec3,
}

pub struct PointLight {
    pub position: Vec3A,
    pub colour_and_intensity: Vec4,
}

pub struct MaterialInfo {
    pub diffuse_texture: u32,
    pub metallic_roughness_texture: i32,
    pub normal_map_texture: i32,
    pub emissive_texture: i32,
    pub fallback_metallic_factor: f32,
    pub fallback_roughness_factor: f32,
}

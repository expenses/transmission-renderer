#![no_std]

use glam::{Mat4, UVec2, Vec2, Vec3A, Vec4};

pub struct PushConstants {
    pub proj_view: Mat4,
    pub view_position: Vec3A,
    pub sun_dir: Vec3A,
    pub sun_intensity: Vec3A,
    pub tile_size_in_pixels: Vec2,
    pub num_tiles: UVec2,
}

pub struct PointLight {
    pub position: Vec3A,
    pub colour_and_intensity: Vec4,
}

#![no_std]

use glam::{Mat4, Vec3A};

pub struct PushConstants {
    pub proj_view: Mat4,
    pub view_position: Vec3A,
    pub sun_dir: Vec3A,
    pub sun_intensity: Vec3A,
}

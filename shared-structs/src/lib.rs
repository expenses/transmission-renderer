#![no_std]

use core::ops::Mul;
use glam::{Mat4, Quat, UVec2, Vec2, Vec3, Vec3A, Vec4};
#[cfg(target_arch = "spirv")]
use num_traits::Float;

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub proj_view: Mat4,
    pub view_position: Vec3A,
    pub tile_size_in_pixels: Vec2,
    pub num_tiles: UVec2,
    pub framebuffer_size: UVec2,
    pub debug_froxels: u32,
    pub ggx_lut_texture_index: u32,
}

pub struct SunUniform {
    pub dir: Vec3A,
    pub intensity: Vec3,
}

pub struct PointLight {
    pub position: Vec3A,
    pub colour_emission_and_falloff_distance: Vec4,
}

impl PointLight {
    pub fn new(position: Vec3, colour: Vec3, intensity: f32) -> Self {
        fn distance_at_strength(intensity: f32, strength: f32) -> f32 {
            (intensity / strength).sqrt()
        }

        let distance_at_0_1 = distance_at_strength(intensity, 0.1);

        Self {
            position: position.into(),
            colour_emission_and_falloff_distance: (colour * intensity).extend(distance_at_0_1)
        }
    }
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
pub struct Textures {
    pub diffuse: i32,
    pub metallic_roughness: i32,
    pub normal_map: i32,
    pub emissive: i32,
    pub occlusion: i32,
    pub transmission: i32,
    pub thickness: i32,
    pub specular: i32,
    pub specular_colour: i32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
pub struct MaterialInfo {
    pub textures: Textures,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_clipping_cutoff: f32,
    pub diffuse_factor: Vec4,
    pub emissive_factor: Vec3A,
    pub normal_map_scale: f32,
    pub occlusion_strength: f32,
    pub index_of_refraction: f32,
    pub transmission_factor: f32,
    pub thickness_factor: f32,
    pub attenuation_distance: f32,
    pub attenuation_colour: Vec3A,
    pub specular_factor: f32,
    pub specular_colour_factor: Vec3A,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
pub struct PackedSimilarity {
    pub translation_and_scale: Vec4,
    pub rotation: Quat,
}

impl PackedSimilarity {
    pub fn unpack(self) -> Similarity {
        Similarity {
            translation: self.translation_and_scale.truncate(),
            scale: self.translation_and_scale.w,
            rotation: self.rotation,
        }
    }
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
pub struct Similarity {
    pub translation: Vec3,
    pub scale: f32,
    pub rotation: Quat,
}

impl Similarity {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        scale: 1.0,
        rotation: Quat::IDENTITY,
    };

    pub fn pack(self) -> PackedSimilarity {
        PackedSimilarity {
            translation_and_scale: self.translation.extend(self.scale),
            rotation: self.rotation,
        }
    }
}

impl Mul<Similarity> for Similarity {
    type Output = Self;

    fn mul(self, child: Self) -> Self {
        Self {
            translation: self * child.translation,
            rotation: self.rotation * child.rotation,
            scale: self.scale * child.scale,
        }
    }
}

impl Mul<Vec3> for Similarity {
    type Output = Vec3;

    fn mul(self, vector: Vec3) -> Vec3 {
        self.translation + (self.scale * (self.rotation * vector))
    }
}

// GPU Culling architecture:
//
// * instances - as usual. References the primitive being drawn.
// * primitive infos - per each primitive being instanced. Also contains the data for drawing the primitives.
// * draw buffers - primitive draw commands are demultiplexed into this.
// * draws - per each potential draw. Has a length equal to the number of primitives.

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
pub struct Instance {
    pub transform: PackedSimilarity,
    pub primitive_id: u32,
    pub material_id: u32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
pub struct PrimitiveInfo {
    pub packed_bounding_sphere: Vec4,
    pub draw_buffer_index: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub first_instance: u32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
pub struct CullingPushConstants {
    pub view: Mat4,
    // As the frustum planes are symmetrical, we only need some of the data:
    // https://github.com/zeux/niagara/commit/6db6db6a7f152fe3b7ba310e267d93d3d7c96ef3
    pub frustum_x_xz: Vec2,
    pub frustum_y_yz: Vec2,
    pub z_near: f32,
}

pub struct Frustum {
    planes: [Vec4; 4],
    z_slice: Vec2,
}

pub const MAX_LIGHTS_PER_FROXEL: u32 = 128;

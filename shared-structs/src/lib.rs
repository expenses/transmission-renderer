#![no_std]

use core::ops::Mul;
use glam::{Mat3, Mat4, Quat, UVec2, Vec2, Vec3, Vec3A, Vec4};
#[cfg(target_arch = "spirv")]
use num_traits::Float;

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PushConstants {
    pub proj_view: Mat4,
    pub view_position: Vec3A,
    pub framebuffer_size: UVec2,
    pub acceleration_structure_address: u64,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Uniforms {
    pub light_clustering_coefficients: LightClusterCoefficients,
    pub sun_dir: Vec3A,
    pub sun_intensity: Vec3A,
    pub tile_size_in_pixels: Vec2,
    pub num_tiles: UVec2,
    pub debug_froxels: u32,
    pub ggx_lut_texture_index: u32,
}

// https://google.github.io/filament/Filament.md.html#imagingpipeline/lightpath/clusteredforwardrendering
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct LightClusterCoefficients {
    pub coefficient_scale: f32,
    pub coefficient_bias: f32,
    pub index_scale: f32,
    pub index_bias: f32,
}

impl LightClusterCoefficients {
    pub fn new(z_near: f32, z_far: f32, z_special_near: f32, max_depth_slices: u32) -> Self {
        Self {
            // todo: filament uses opengl depth (-1 to 1) as opposed to vulkan depth (0 to 1)
            // so we set the scale and bias accordingly
            coefficient_scale: 2.0 * ((z_far / z_near) - 1.0),
            coefficient_bias: 1.0,
            index_scale: ((max_depth_slices - 1) as f32 / (z_special_near / z_far).log2()),
            index_bias: max_depth_slices as f32
        }
    }

    // https://www.desmos.com/calculator/spahzn1han
    pub fn get_depth_slice(self, frag_depth: f32) -> u32 {
        let linear_depth = frag_depth * self.coefficient_scale + self.coefficient_bias;
        (linear_depth.log2() * self.index_scale + self.index_bias).max(0.0) as u32
    }
}

#[repr(C)]
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
            colour_emission_and_falloff_distance: (colour * intensity).extend(distance_at_0_1),
        }
    }
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[repr(C)]
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
#[repr(C)]
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
#[repr(C)]
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
#[repr(C)]
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

    pub fn as_mat4(self) -> Mat4 {
        Mat4::from_translation(self.translation)
            * Mat4::from_mat3(Mat3::from_quat(self.rotation))
            * Mat4::from_scale(Vec3::splat(self.scale))
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
#[repr(C)]
pub struct Instance {
    pub transform: PackedSimilarity,
    pub primitive_id: u32,
    pub material_id: u32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PrimitiveInfo {
    pub packed_bounding_sphere: Vec4,
    pub draw_buffer_index: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub first_instance: u32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CullingPushConstants {
    pub view: Mat4,
    // As the frustum planes are symmetrical, we only need some of the data:
    // https://github.com/zeux/niagara/commit/6db6db6a7f152fe3b7ba310e267d93d3d7c96ef3
    pub frustum_x_xz: Vec2,
    pub frustum_y_yz: Vec2,
    pub z_near: f32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FroxelData {
    pub left_plane: Plane,
    pub right_plane: Plane,
    pub top_plane: Plane,
    pub bottom_plane: Plane,
    pub z_near: f32,
    pub z_far: f32,
}

impl FroxelData {
    fn contains_sphere(&self, mut center: Vec3, radius: f32, view: Mat4) -> bool {
        center = (view * center.extend(1.0)).truncate();
        center.z = -center.z;

        let mut visible = 1;

        visible &= (center.z + radius > self.z_near) as u32;
        visible &= (center.z - radius <= self.z_far) as u32;

        visible &= (self.left_plane.signed_distance(center) < radius) as u32;
        visible &= (self.right_plane.signed_distance(center) < radius) as u32;
        visible &= (self.top_plane.signed_distance(center) < radius) as u32;
        visible &= (self.bottom_plane.signed_distance(center) < radius) as u32;

        visible != 0
    }
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Plane {
    inner: Vec3A,
}

impl Plane {
    pub fn new(plane: Vec3) -> Self {
        Self {
            inner: plane.into(),
        }
    }

    fn signed_distance(&self, point: Vec3) -> f32 {
        Vec3::from(self.inner).dot(point)
    }
}

pub const MAX_LIGHTS_PER_FROXEL: u32 = 128;

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct AccelerationStructureDebuggingUniforms {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub size: UVec2,
}

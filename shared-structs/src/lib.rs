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
    pub cluster_size_in_pixels: Vec2,
    pub num_clusters: UVec2,
    pub debug_clusters: u32,
    pub ggx_lut_texture_index: u32,
}

// https://google.github.io/filament/Filament.md.html#imagingpipeline/lightpath/clusteredforwardrendering
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct LightClusterCoefficients {
    pub z_near: f32,
    pub z_far: f32,
    pub scale: f32,
    pub bias: f32,
    pub num_depth_slices: u32,
}

impl LightClusterCoefficients {
    pub fn new(z_near: f32, z_far: f32, num_depth_slices: u32) -> Self {
        Self {
            z_near,
            z_far,
            num_depth_slices,
            scale: num_depth_slices as f32 / (z_far / z_near).log2(),
            bias: -(num_depth_slices as f32 * z_near.log2() / (z_far / z_near).log2()),
        }
    }

    fn linear_depth(self, frag_depth: f32) -> f32 {
        let depth_range = 2.0 * (1.0 - frag_depth) - 1.0;
        2.0 * self.z_near * self.z_far
            / (self.z_far + self.z_near - depth_range * (self.z_far - self.z_near))
    }

    // https://www.desmos.com/calculator/spahzn1han
    pub fn get_depth_slice(self, frag_depth: f32) -> u32 {
        (self.linear_depth(frag_depth).log2() * self.scale + self.bias).max(0.0) as u32
    }

    pub fn slice_to_depth(self, slice: u32) -> f32 {
        -self.z_near * (self.z_far / self.z_near).powf(slice as f32 / self.num_depth_slices as f32)
    }
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
// Really messily packed together :sweat_smile:
pub struct Light {
    pub position_and_spotlight_epsilon: Vec4,
    pub colour_emission_and_falloff_distance_sq: Vec4,
    pub spotlight_direction_and_outer_angle: Vec4,
}

impl Light {
    pub fn set_spotlight_direction(&mut self, direction: Vec3) {
        let outer_angle = self.spotlight_direction_and_outer_angle.w;
        self.spotlight_direction_and_outer_angle = direction.extend(outer_angle);
    }

    fn distance_sq_at_strength(intensity: f32, strength: f32) -> f32 {
        intensity / strength
    }

    pub fn position(self) -> Vec3 {
        self.position_and_spotlight_epsilon.truncate()
    }

    pub fn new_point(position: Vec3, colour: Vec3, intensity: f32) -> Self {
        let distance_sq_at_0_05 = Self::distance_sq_at_strength(intensity, 0.05);

        Self {
            position_and_spotlight_epsilon: position.extend(0.0),
            colour_emission_and_falloff_distance_sq: (colour * intensity)
                .extend(distance_sq_at_0_05),
            spotlight_direction_and_outer_angle: Vec4::ZERO,
        }
    }

    pub fn new_spot(
        position: Vec3,
        colour: Vec3,
        intensity: f32,
        direction: Vec3,
        inner_angle_rad: f32,
        outer_angle_rad: f32,
    ) -> Self {
        let distance_sq_at_0_05 = Self::distance_sq_at_strength(intensity, 0.05);

        let epsilon = inner_angle_rad.cos() - outer_angle_rad.cos();

        Self {
            position_and_spotlight_epsilon: position.extend(epsilon),
            colour_emission_and_falloff_distance_sq: (colour * intensity)
                .extend(distance_sq_at_0_05),
            spotlight_direction_and_outer_angle: direction.extend(outer_angle_rad),
        }
    }

    pub fn is_a_spotlight(self) -> bool {
        self.spotlight_direction_and_outer_angle.w != 0.0
    }

    pub fn spotlight_factor(self, direction_to_light: Vec3) -> f32 {
        let spotlight_direction = self.spotlight_direction_and_outer_angle.truncate();

        let theta = (-direction_to_light).dot(spotlight_direction);

        let outer_angle = self.spotlight_direction_and_outer_angle.w;
        let epsilon = self.position_and_spotlight_epsilon.w;

        ((theta - outer_angle.cos()) / epsilon).max(0.0)
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
pub struct ClusterAabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl ClusterAabb {
    pub fn distance_sq(self, point: Vec3) -> f32 {
        // Evaluate the distance on each axis
        let distances = (Vec3::from(self.min) - point)
            .max(point - Vec3::from(self.max))
            .max(Vec3::ZERO);

        distances.length_squared()
    }

    // https://simoncoenen.com/blog/programming/graphics/SpotlightCulling
    pub fn cull_spotlight(self, origin: Vec3, direction: Vec3, angle: f32, range: f32) -> bool {
        let center = (self.min + self.max) / 2.0;
        let radius = self.max.distance(center);

        let vector = Vec3::from(center) - origin;

        let vector_len_sq = vector.dot(vector);

        let vector_1_len = vector.dot(direction);
        let vector_1_len_sq = vector_1_len * vector_1_len;

        let distance_closest_point = angle.cos() * (vector_len_sq - vector_1_len_sq).sqrt() - vector_1_len * angle.sin();

        let mut cull = 0;

        // angle cull
        cull |= (distance_closest_point > radius) as u32;
        // front cull
        cull |= (vector_1_len > radius + range) as u32;
        // back cull
        cull |= (vector_1_len < -radius) as u32;

        cull == 1
    }
}

pub const MAX_LIGHTS_PER_CLUSTER: u32 = 128;

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct AccelerationStructureDebuggingUniforms {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub size: UVec2,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct WriteClusterDataPushConstants {
    pub inverse_perspective: Mat4,
    pub screen_dimensions: UVec2,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct AssignLightsPushConstants {
    pub view_matrix: Mat4,
    pub view_rotation: Quat,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Particle {
    pub position: Vec3A,
    pub start_frame: u32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ParticleSimPushConstants {
    pub current_frame: u32,
    pub delta_time: f32,
}

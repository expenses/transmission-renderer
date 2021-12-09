use crate::{index, TextureSampler, noise::BlueNoiseSampler};
use glam_pbr::{
    basic_brdf, light_direction_and_attenuation, BasicBrdfParams, BrdfResult, IndexOfRefraction,
    Light as LightDir, MaterialParams, Normal, PerceptualRoughness, View,
};
use shared_structs::{Light, MaterialInfo, Uniforms};
use spirv_std::{
    glam::{Mat3, Vec2, Vec3, Vec4, Vec4Swizzles},
    num_traits::Float,
    ray_tracing::{AccelerationStructure, CommittedIntersection, RayFlags, RayQuery},
};

pub fn evaluate_lights_transmission(
    material_params: MaterialParams,
    view: View,
    position: Vec3,
    normal: Normal,
    uniforms: &Uniforms,
    light_params: LightParams,
    #[cfg(target_feature = "RayQueryKHR")] acceleration_structure: &AccelerationStructure,
) -> (BrdfResult, Vec3) {
    #[cfg(target_feature = "RayQueryKHR")]
    spirv_std::ray_query!(let mut shadow_ray);

    #[cfg(target_feature = "RayQueryKHR")]
    let factor = trace_shadow_ray(
        shadow_ray,
        acceleration_structure,
        position,
        uniforms.sun_dir.into(),
        10_000.0,
    );

    #[cfg(not(target_feature = "RayQueryKHR"))]
    let factor = 1.0;

    let sun_intensity = Vec3::from(uniforms.sun_intensity) * factor;

    let mut sum = basic_brdf(BasicBrdfParams {
        light: LightDir(uniforms.sun_dir.into()),
        light_intensity: sun_intensity,
        normal,
        view,
        material_params,
    });

    let mut transmission = sun_intensity
        * glam_pbr::transmission_btdf(
            material_params,
            normal,
            view,
            LightDir(uniforms.sun_dir.into()),
        );

    let mut current_light = light_params.light_indices_offset;
    let end = light_params.end();

    while current_light < end {
        let light = light_params.index(current_light);

        let (direction, distance, attenuation) =
            light_direction_and_attenuation(position, light.position());

        #[cfg(target_feature = "RayQueryKHR")]
        let factor = trace_shadow_ray(
            shadow_ray,
            acceleration_structure,
            position,
            direction,
            distance,
        );

        #[cfg(not(target_feature = "RayQueryKHR"))]
        let factor = 1.0;

        let light_emission = light.colour_emission_and_falloff_distance_sq.truncate() * factor;

        sum = sum
            + basic_brdf(BasicBrdfParams {
                light: LightDir(direction),
                light_intensity: light_emission * attenuation,
                normal,
                view,
                material_params,
            });

        transmission += light_emission
            * attenuation
            * glam_pbr::transmission_btdf(material_params, normal, view, LightDir(direction));

        current_light += 1;
    }

    (sum, transmission)
}

pub(crate) fn trace_shadow_ray(
    ray: &mut RayQuery,
    acceleration_structure: &AccelerationStructure,
    origin: Vec3,
    direction: Vec3,
    max_t: f32,
) -> f32 {
    unsafe {
        ray.initialize(
            acceleration_structure,
            RayFlags::NONE,
            0xff,
            origin,
            0.001,
            direction,
            max_t,
        );

        while ray.proceed() {
            let material_index = ray.get_candidate_intersection_instance_custom_index();
            ray.confirm_intersection();
        }

        match ray.get_committed_intersection_type() {
            CommittedIntersection::None => 1.0,
            _ => 0.0,
        }
    }
}

pub struct LightParams<'a> {
    pub num_lights: u32,
    pub light_indices_offset: u32,
    pub light_indices: &'a [u32],
    pub lights: &'a [Light],
}

impl<'a> LightParams<'a> {
    fn index(&'a self, id: u32) -> &'a Light {
        let light_index = *index(self.light_indices, id);
        index(self.lights, light_index)
    }

    fn end(&self) -> u32 {
        self.light_indices_offset + self.num_lights
    }
}

pub fn evaluate_lights(
    material_params: MaterialParams,
    view: View,
    position: Vec3,
    normal: Normal,
    uniforms: &Uniforms,
    light_params: LightParams,
    #[cfg(target_feature = "RayQueryKHR")] acceleration_structure: &AccelerationStructure,
    #[cfg(target_feature = "RayQueryKHR")] blue_noise_sampler: &mut BlueNoiseSampler,
    sun_shadow_value: f32,
) -> BrdfResult {
    let factor = sun_shadow_value
        // todo: ambient lighting via probes or idk!
        .max(0.1);

    let mut sum = basic_brdf(BasicBrdfParams {
        light: LightDir(uniforms.sun_dir.into()),
        light_intensity: Vec3::from(uniforms.sun_intensity) * factor,
        normal,
        view,
        material_params,
    });

    #[cfg(target_feature = "RayQueryKHR")]
    spirv_std::ray_query!(let mut shadow_ray);

    let mut current_light = light_params.light_indices_offset;
    let end = light_params.end();

    while current_light < end {
        let light = light_params.index(current_light);

        let (direction, distance, attenuation) =
            light_direction_and_attenuation(position, light.position());

        let mut factor = 1.0;

        #[cfg(target_feature = "RayQueryKHR")]
        {
            factor *= trace_shadow_ray(
                shadow_ray,
                acceleration_structure,
                position,
                direction,
                distance,
            );
        }

        if light.is_a_spotlight() {
            factor *= light.spotlight_factor(direction);
        }

        let light_emission = light.colour_emission_and_falloff_distance_sq.truncate() * factor;

        sum = sum
            + basic_brdf(BasicBrdfParams {
                light: LightDir(direction),
                light_intensity: light_emission * attenuation,
                normal,
                view,
                material_params,
            });

        current_light += 1;
    }

    sum
}

pub(crate) fn calculate_normal(
    interpolated_normal: Vec3,
    texture_sampler: &TextureSampler,
    material: &MaterialInfo,
    view_vector: Vec3,
    uv: Vec2,
) -> Normal {
    let mut normal = interpolated_normal.normalize();

    if material.textures.normal_map != -1 {
        let map_normal = texture_sampler
            .sample(material.textures.normal_map as u32)
            .truncate();
        let map_normal = map_normal * 255.0 / 127.0 - 128.0 / 127.0;

        normal = (compute_cotangent_frame(normal, -view_vector, uv) * map_normal).normalize();
    };

    Normal(normal)
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

pub(crate) fn get_material_params(
    diffuse: Vec4,
    material: &MaterialInfo,
    texture_sampler: &TextureSampler,
) -> MaterialParams {
    let mut metallic = material.metallic_factor;
    let mut roughness = material.roughness_factor;

    if material.textures.metallic_roughness != -1 {
        let sample = texture_sampler.sample(material.textures.metallic_roughness as u32);

        // These two are switched!
        let metallic_roughness = sample.zy();

        metallic *= metallic_roughness.x;
        roughness *= metallic_roughness.y;
    }

    let mut specular_colour = Vec3::from(material.specular_colour_factor);

    if material.textures.specular_colour != -1 {
        let sample = texture_sampler.sample(material.textures.specular_colour as u32);
        specular_colour *= sample.truncate();
    }

    let mut specular_factor = material.specular_factor;

    if material.textures.specular != -1 {
        let sample = texture_sampler.sample(material.textures.specular as u32);
        specular_factor *= sample.w;
    }

    MaterialParams {
        diffuse_colour: diffuse.truncate(),
        metallic,
        perceptual_roughness: PerceptualRoughness(roughness),
        index_of_refraction: IndexOfRefraction(material.index_of_refraction),
        specular_colour,
        specular_factor,
    }
}

pub(crate) fn get_emission(material: &MaterialInfo, texture_sampler: &TextureSampler) -> Vec3 {
    let mut emission = Vec3::from(material.emissive_factor);

    if material.textures.emissive != -1 {
        emission *= texture_sampler
            .sample(material.textures.emissive as u32)
            .truncate();
    }

    emission
}

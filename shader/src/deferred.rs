use crate::*;

struct GBuffer<'a> {
    position: &'a Image!(2D, type=f32, sampled),
    normal: &'a Image!(2D, type=f32, sampled),
    uv: &'a Image!(2D, type=f32, sampled),
    material: &'a Image!(2D, type=u32, sampled),
}

impl<'a> GBuffer<'a> {
    fn sample(&self, sampler: Sampler, coord: Vec2) -> GBufferSample {
        let position: Vec4 = self.position.sample(sampler, coord);
        let normal: Vec4 = self.normal.sample(sampler, coord);
        let uv: Vec4 = self.uv.sample(sampler, coord);
        let material_id: UVec4 = self.material.sample(sampler, coord);

        GBufferSample {
            position: position.truncate(),
            frag_coord_z: position.w,
            normal: normal.truncate(),
            uv: uv.xy(),
            material_id: material_id.x,
        }
    }
}

struct GBufferSample {
    position: Vec3,
    frag_coord_z: f32,
    normal: Vec3,
    uv: Vec2,
    material_id: u32,
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn defer_opaque(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(frag_coord)] frag_coord: Vec4,
    out_position: &mut Vec4,
    out_normal: &mut Vec4,
    out_uv: &mut Vec2,
    #[spirv(flat)] out_material: &mut u32,
) {
    // todo: could just read the depth buffer instead.
    *out_position = position.extend(frag_coord.z);
    *out_normal = normal.extend(1.0);
    *out_uv = uv;
    *out_material = material_id;
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn defer_alpha_clip(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    out_position: &mut Vec4,
    out_normal: &mut Vec4,
    out_uv: &mut Vec2,
    #[spirv(flat)] out_material: &mut u32,
) {
    let material = index(materials, material_id);

    let texture_sampler = TextureSampler {
        uv,
        textures,
        sampler: *sampler,
    };

    let mut diffuse = material.diffuse_factor;

    if material.textures.diffuse != -1 {
        diffuse *= texture_sampler.sample(material.textures.diffuse as u32)
    }

    if diffuse.w < material.alpha_clipping_cutoff {
        spirv_std::arch::kill();
    }

    *out_position = position.extend(frag_coord.z);
    *out_normal = normal.extend(1.0);
    *out_uv = uv;
    *out_material = material_id;
}

#[spirv(fragment)]
pub fn render(
    uv: Vec2,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 8)] nearest_sampler: &Sampler,

    #[spirv(descriptor_set = 1, binding = 0)] position_buffer: &Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 1, binding = 1)] normal_buffer: &Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 1, binding = 2)] uv_buffer: &Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 1, binding = 3)] material_buffer: &Image!(2D, type=u32, sampled),

    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] lights: &[Light],
    #[spirv(descriptor_set = 2, binding = 1, storage_buffer)] cluster_light_counts: &[u32],
    #[spirv(descriptor_set = 2, binding = 2, storage_buffer)] light_indices: &[u32],
    #[spirv(descriptor_set = 3, binding = 0)] sun_shadow_buffer: &Image!(2D, type=f32, sampled),
    #[spirv(frag_coord)] frag_coord: Vec4,
    hdr_framebuffer: &mut Vec4,
    opaque_sampled_framebuffer: &mut Vec4,
) {
    let g_buffer = GBuffer {
        position: position_buffer,
        normal: normal_buffer,
        uv: uv_buffer,
        material: material_buffer,
    };

    let GBufferSample {
        position,
        normal,
        uv,
        material_id,
        frag_coord_z,
    } = g_buffer.sample(*nearest_sampler, uv);

    let material = index(materials, material_id);

    let texture_sampler = TextureSampler {
        uv,
        textures,
        sampler: *sampler,
    };

    let mut diffuse = material.diffuse_factor;

    if material.textures.diffuse != -1 {
        diffuse *= texture_sampler.sample(material.textures.diffuse as u32)
    }

    let view_vector = Vec3::from(push_constants.view_position) - position;
    let view = View(view_vector.normalize());

    let normal = calculate_normal(normal, &texture_sampler, material, view_vector, uv);

    let material_params = get_material_params(diffuse, material, &texture_sampler);

    let emission = get_emission(material, &texture_sampler);

    let cluster_z = uniforms
        .light_clustering_coefficients
        .get_depth_slice(frag_coord_z);

    let cluster = {
        let cluster_xy = (frag_coord.xy() / uniforms.cluster_size_in_pixels).as_uvec2();

        cluster_z * uniforms.num_clusters.x * uniforms.num_clusters.y
            + cluster_xy.y * uniforms.num_clusters.x
            + cluster_xy.x
    };

    #[cfg(target_feature = "RayQueryKHR")]
    let acceleration_structure =
        unsafe { AccelerationStructure::from_u64(push_constants.acceleration_structure_address) };

    let mut blue_noise_sampler = BlueNoiseSampler {
        textures,
        sampler: *sampler,
        uniforms,
        frag_coord: frag_coord.xy(),
        iteration: 0,
    };

    let sun_shadow_value = {
        let uv = frag_coord.xy() / push_constants.framebuffer_size.as_vec2();
        let output: Vec4 = sun_shadow_buffer.sample(*sampler, uv);
        output.x
    };

    let num_lights = *index(cluster_light_counts, cluster);

    let result = evaluate_lights(
        material_params,
        view,
        position,
        normal,
        uniforms,
        LightParams {
            num_lights,
            light_indices_offset: cluster * MAX_LIGHTS_PER_CLUSTER,
            light_indices,
            lights,
        },
        #[cfg(target_feature = "RayQueryKHR")]
        &acceleration_structure,
        #[cfg(target_feature = "RayQueryKHR")]
        &mut blue_noise_sampler,
        sun_shadow_value,
    );

    let mut output = (result.diffuse + result.specular + emission).extend(1.0);

    if uniforms.debug_clusters != 0 {
        output = (debug_colour_for_id(num_lights) + (debug_colour_for_id(cluster) - 0.5) * 0.025)
            .extend(1.0);
        //output = (debug_colour_for_id(cluster_z)).extend(1.0);
    }

    //output = Vec4::new(sun_shadow_value, sun_shadow_value, sun_shadow_value, 1.0);

    *hdr_framebuffer = output;
    *opaque_sampled_framebuffer = output;
}

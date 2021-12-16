use crate::*;

#[spirv(fragment)]
pub fn transmission(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(flat)] model_scale: f32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 4)] clamp_sampler: &Sampler,
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] lights: &[Light],
    #[spirv(descriptor_set = 2, binding = 1, storage_buffer)] cluster_light_counts: &[u32],
    #[spirv(descriptor_set = 2, binding = 2, storage_buffer)] cluster_light_indices: &[u32],
    #[spirv(descriptor_set = 3, binding = 0)] framebuffer: &Image!(2D, type=f32, sampled),
    #[spirv(frag_coord)] frag_coord: Vec4,
    output: &mut Vec4,
) {
    let material = index(materials, material_id);

    let texture_sampler = TextureSampler {
        uv,
        textures,
        sampler: *sampler,
    };

    let mut diffuse = material.diffuse_factor;

    if material.textures.diffuse != -1 {
        diffuse *= texture_sampler.sample(material.textures.diffuse as u32);
    }

    let mut transmission_factor = material.transmission_factor;

    if material.textures.transmission != -1 {
        transmission_factor *= texture_sampler
            .sample(material.textures.transmission as u32)
            .x;
    }

    let view_vector = Vec3::from(push_constants.view_position) - position;
    let view = View(view_vector.normalize());

    let normal = calculate_normal(normal, &texture_sampler, material, view_vector, uv);

    let material_params = get_material_params(diffuse, material, &texture_sampler);

    let emission = get_emission(material, &texture_sampler);

    let cluster = {
        let cluster_xy = (frag_coord.xy() / uniforms.cluster_size_in_pixels).as_uvec2();

        let cluster_z = uniforms
            .light_clustering_coefficients
            .get_depth_slice(frag_coord.z);

        cluster_z * uniforms.num_clusters.x * uniforms.num_clusters.y
            + cluster_xy.y * uniforms.num_clusters.x
            + cluster_xy.x
    };

    #[cfg(target_feature = "RayQueryKHR")]
    let acceleration_structure =
        unsafe { AccelerationStructure::from_u64(push_constants.acceleration_structure_address) };

    let (result, mut transmission) = evaluate_lights_transmission(
        material_params,
        view,
        position,
        normal,
        uniforms,
        LightParams {
            num_lights: *index(cluster_light_counts, cluster),
            light_indices_offset: cluster * MAX_LIGHTS_PER_CLUSTER,
            light_indices: cluster_light_indices,
            lights,
        },
        #[cfg(target_feature = "RayQueryKHR")]
        &acceleration_structure,
    );

    let mut thickness = material.thickness_factor;

    if material.textures.thickness != -1 {
        thickness *= texture_sampler.sample(material.textures.thickness as u32).y;
    }

    let ggx_lut_sampler = |normal_dot_view: f32, perceptual_roughness: PerceptualRoughness| {
        let uv = Vec2::new(normal_dot_view, perceptual_roughness.0);

        let texture = unsafe { textures.index(uniforms.ggx_lut_texture_index as usize) };
        let sample: Vec4 = texture.sample(*clamp_sampler, uv);

        sample.xy()
    };

    let framebuffer_sampler = |uv, lod| {
        let sample: Vec4 = framebuffer.sample_by_lod(*clamp_sampler, uv, lod);
        sample.truncate()
    };

    transmission += ibl_volume_refraction(
        IblVolumeRefractionParams {
            proj_view_matrix: push_constants.proj_view,
            position,
            material_params,
            framebuffer_size_x: push_constants.framebuffer_size.x,
            normal,
            view,
            thickness,
            model_scale,
            attenuation_distance: material.attenuation_distance,
            attenuation_colour: material.attenuation_colour.into(),
        },
        framebuffer_sampler,
        ggx_lut_sampler,
    );

    let real_transmission = transmission_factor * transmission;

    let diffuse = result.diffuse.lerp(real_transmission, transmission_factor);

    *output = (diffuse + result.specular + emission).extend(1.0);
}

#[spirv(fragment)]
pub fn opaque(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] lights: &[Light],
    #[spirv(descriptor_set = 2, binding = 1, storage_buffer)] cluster_light_counts: &[u32],
    #[spirv(descriptor_set = 2, binding = 2, storage_buffer)] cluster_light_indices: &[u32],
    #[spirv(descriptor_set = 3, binding = 0)] sun_shadow_buffer: &Image!(2D, format=rgba8, sampled=false),
    #[spirv(descriptor_set = 4, binding = 7)] reprojection_results: &Image!(2D, format=rgba8, sampled=false),
    #[spirv(descriptor_set = 4, binding = 9)] current_moments: &Image!(2D, format=rgba8, sampled=false),
    #[spirv(frag_coord)] frag_coord: Vec4,
    hdr_framebuffer: &mut Vec4,
    opaque_sampled_framebuffer: &mut Vec4,
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

    let view_vector = Vec3::from(push_constants.view_position) - position;
    let view = View(view_vector.normalize());

    let normal = calculate_normal(normal, &texture_sampler, material, view_vector, uv);

    let material_params = get_material_params(diffuse, material, &texture_sampler);

    let emission = get_emission(material, &texture_sampler);

    let cluster_z = uniforms
        .light_clustering_coefficients
        .get_depth_slice(frag_coord.z);

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

    let num_lights = *index(cluster_light_counts, cluster);

    #[cfg(target_feature = "RayQueryKHR")]
    let sun_shadow_value = unsafe {
        let sample: Vec4 = sun_shadow_buffer.read(frag_coord.xy().as_uvec2());
        sample.x
    };

    #[cfg(not(target_feature = "RayQueryKHR"))]
    let sun_shadow_value = 1.0;

    let result = evaluate_lights(
        material_params,
        view,
        position,
        normal,
        uniforms,
        LightParams {
            num_lights,
            light_indices_offset: cluster * MAX_LIGHTS_PER_CLUSTER,
            light_indices: cluster_light_indices,
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

        output = unsafe {
            let sample: Vec2 = reprojection_results.read(frag_coord.xy().as_uvec2());
            sample
        }
        .extend(0.0)
        .extend(1.0);

        /*
        output = unsafe {
            let sample: Vec3 = current_moments.read(frag_coord.xy().as_uvec2());
            sample
        }.extend(1.0);
        */
    }

    //output = Vec4::new(sun_shadow_value, sun_shadow_value, sun_shadow_value, 1.0);

    *hdr_framebuffer = output;
    *opaque_sampled_framebuffer = output;
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn tonemap(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] texture: &Image!(2D, type=f32, sampled),
    #[spirv(push_constant)] params: &BakedLottesTonemapperParams,
    output: &mut Vec4,
) {
    let sample: Vec4 = texture.sample(*sampler, uv);

    *output = LottesTonemapper
        .tonemap(sample.truncate(), *params)
        .extend(1.0);

    //*output = sample;
}

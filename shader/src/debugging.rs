use crate::*;

#[cfg(target_feature = "RayQueryKHR")]
#[spirv(compute(threads(8, 8)))]
pub fn acceleration_structure_debugging(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push_constants: &PushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    #[spirv(descriptor_set = 0, binding = 5, storage_buffer)] indices: &[u32],
    #[spirv(descriptor_set = 0, binding = 6, storage_buffer)] uvs: &[Vec2],
    #[spirv(descriptor_set = 0, binding = 7, storage_buffer)] primitives: &[PrimitiveInfo],
    #[spirv(descriptor_set = 1, binding = 0)] output: &Image!(2D, format=rgba16f, sampled=false),
    #[spirv(descriptor_set = 1, binding = 1, uniform)] uniforms: &AccelerationStructureDebuggingUniforms,
    #[spirv(descriptor_set = 2, binding = 0, storage_buffer)] instances: &[Instance],
) {
    let id_xy = id.truncate();

    let pixel_center = id_xy.as_vec2() + 0.5;

    let texture_coordinates = pixel_center / uniforms.size.as_vec2();

    let render_coordinates = texture_coordinates * 2.0 - 1.0;

    // Transform [0, 0, 0] in view-space into world space
    let origin = (uniforms.view_inverse * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
    // Transform the render coordinates into project-y space
    let target =
        uniforms.proj_inverse * Vec4::new(render_coordinates.x, render_coordinates.y, 1.0, 1.0);
    let local_direction_vector = target.truncate().normalize();
    // Rotate the location direction vector into a global direction vector.
    let direction = (uniforms.view_inverse * local_direction_vector.extend(0.0)).truncate();

    let acceleration_structure =
        unsafe { AccelerationStructure::from_u64(push_constants.acceleration_structure_address) };

    use spirv_std::ray_tracing::{
        AccelerationStructure, CommittedIntersection, RayFlags, RayQuery,
    };

    let model_buffers = ModelBuffers {
        indices, uvs
    };

    spirv_std::ray_query!(let mut ray);

    unsafe {
        ray.initialize(
            &acceleration_structure,
            RayFlags::NONE,
            0xff,
            origin,
            0.01,
            direction,
            1000.0,
        );

        let mut colour = Vec3::ZERO;

        while ray.proceed() {
            let instance_id = ray.get_candidate_intersection_instance_custom_index();
            let instance = index(instances, instance_id);
            let material = index(materials, instance.material_id);
            let primitive_info = index(primitives, instance.primitive_id);

            let triangle_index = ray.get_candidate_intersection_primitive_index();
            let indices = model_buffers.get_indices_for_primitive(triangle_index + primitive_info.first_index / 3);
            let barycentric_coords = barycentric_coords_from_hits(ray.get_candidate_intersection_barycentrics());
            let interpolated_uv = model_buffers.interpolate_uv(indices, barycentric_coords);

            let texture_sampler = TextureSampler {
                uv: interpolated_uv,
                textures,
                sampler: *sampler,
            };

            let mut diffuse = material.diffuse_factor;

            if material.textures.diffuse != -1 {
                diffuse *= texture_sampler.sample_by_lod_0(material.textures.diffuse as u32)
            }

            if diffuse.w >= material.alpha_clipping_cutoff {
                ray.confirm_intersection();

                colour = diffuse.truncate();
            }
        }

        /*
        let colour = match ray.get_committed_intersection_type() {
            CommittedIntersection::None => Vec3::ZERO,
            _ => {
                barycentric_coords_from_hits(ray.get_committed_intersection_barycentrics())
            }
        };
        */

        output.write(id.truncate(), colour);
    };
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn cluster_debugging_vs(
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] cluster_data: &[ClusterAabb],
    #[spirv(vertex_index)] vertex_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_colour: &mut Vec3,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let cluster = index(cluster_data, vertex_index / 8);

    let points = [
        Vec3::from(cluster.min),
        Vec3::new(cluster.min.x, cluster.min.y, cluster.max.z),
        Vec3::from(cluster.min),
        Vec3::new(cluster.min.x, cluster.max.y, cluster.min.z),
        Vec3::from(cluster.min),
        Vec3::new(cluster.max.x, cluster.min.y, cluster.min.z),
        Vec3::new(cluster.max.x, cluster.max.y, cluster.min.z),
        Vec3::from(cluster.max),
    ];

    let position = *index(&points, vertex_index % points.len() as u32);

    *builtin_pos = push_constants.proj_view * position.extend(1.0);

    *out_colour = Vec3::X.lerp(Vec3::Y, (vertex_index % 2) as f32);
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn cluster_debugging_fs(
    colour: Vec3,
    hdr_framebuffer: &mut Vec4,
    opaque_sampled_framebuffer: &mut Vec4,
) {
    let output = colour.extend(1.0);
    *hdr_framebuffer = output;
    *opaque_sampled_framebuffer = output;
}

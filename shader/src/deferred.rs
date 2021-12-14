use crate::*;

// The hlsl code from https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
// is so cursed holy shit. `( v.xy >= 0.0 ? 1.0 : -1.0 )` apparently does a `OpFOrdGreaterThanEqual`
// then a `OpSelect` but there's no way to tell by looking.
fn octahedron_wrap(xy: Vec2) -> Vec2 {
    (1.0 - Vec2::new(xy.y.abs(), xy.x.abs()))
        * Vec2::select(xy.cmpge(Vec2::ZERO), Vec2::ONE, -Vec2::ONE)
}

// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
fn encode_normal_as_octahedron(normal: Vec3) -> Vec2 {
    let normal = normal / (normal.x.abs() + normal.y.abs() + normal.z.abs());

    let mut xy = normal.truncate();

    xy = if normal.z >= 0.0 {
        xy
    } else {
        octahedron_wrap(xy)
    };

    xy * 0.5 + 0.5
}

fn decode_octahedron_as_normal(octahedron: Vec2) -> Vec3 {
    let octahedron = octahedron * 2.0 - 1.0;

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    let normal = Vec3::new(
        octahedron.x,
        octahedron.y,
        1.0 - octahedron.x.abs() - octahedron.y.abs(),
    );
    let t = (-normal.z).max(0.0).min(1.0);

    let mut xy = normal.truncate();
    xy += Vec2::select(xy.cmpge(Vec2::ZERO), Vec2::splat(-t), Vec2::splat(t));
    xy.extend(normal.z).normalize()
}

// https://github.com/GPUOpen-Effects/FidelityFX-Denoiser/issues/1#issue-894257263
fn compute_velocity(clip_position: Vec4, prev_clip_position: Vec4) -> Vec2 {
    (clip_position.xy() / clip_position.w) - (prev_clip_position.xy() / prev_clip_position.w)
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn defer_opaque(
    clip_position: Vec4,
    prev_clip_position: Vec4,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(frag_coord)] frag_coord: Vec4,
    out_normal_and_velocity: &mut Vec4,
) {
    *out_normal_and_velocity = Vec4::from((
        encode_normal_as_octahedron(normal),
        compute_velocity(clip_position, prev_clip_position),
    ));
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(fragment)]
pub fn defer_alpha_clip(
    clip_position: Vec4,
    prev_clip_position: Vec4,
    normal: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_id: u32,
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] materials: &[MaterialInfo],
    out_normal_and_velocity: &mut Vec4,
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

    *out_normal_and_velocity = Vec4::from((
        encode_normal_as_octahedron(normal),
        compute_velocity(clip_position, prev_clip_position),
    ));
}

#[cfg(not(target_feature = "RayQueryKHR"))]
#[spirv(vertex)]
pub fn vs(
    vertex_position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 3, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 1, binding = 0, storage_buffer)] instances: &[Instance],
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(push_constant)] push_constants: &PushConstants,
    out_clip_position: &mut Vec4,
    out_prev_clip_position: &mut Vec4,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
    #[spirv(flat)] out_material_id: &mut u32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let instance = index(instances, instance_index);
    let transform = instance.transform.unpack();

    let position = transform * vertex_position;
    let prev_position = instance.prev_transform.unpack() * vertex_position;

    *out_normal = transform.rotation * normal;
    *out_uv = uv;
    *out_material_id = instance.material_id;

    let clip_position = push_constants.proj_view * position.extend(1.0);
    let prev_clip_position = uniforms.prev_proj_view * prev_position.extend(1.0);

    *builtin_pos = clip_position;
    *out_clip_position = clip_position;
    *out_prev_clip_position = prev_clip_position;
}

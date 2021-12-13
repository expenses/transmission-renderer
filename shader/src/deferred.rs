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
) {
    // todo: could just read the depth buffer instead.
    *out_position = position.extend(frag_coord.z);
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
}

use crate::{mip_levels_for_size, ModelStagingBuffers};
use ash::vk;
use glam::{Quat, Vec2, Vec3};
use shared_structs::{DrawCounts, Instance, Similarity};
use std::path::PathBuf;

pub(crate) fn load_gltf(
    name: &str,
    init_resources: &mut ash_abstractions::InitResources,
    image_manager: &mut ImageManager,
    buffers_to_cleanup: &mut Vec<ash_abstractions::Buffer>,
    model_buffers: &mut ModelStagingBuffers,
    max_draw_counts: &mut DrawCounts,
    base_transform: Similarity,
) -> anyhow::Result<()> {
    let _span = tracy_client::span!(name);

    let importing_gltf_span = tracy_client::span!("Importing gltf");

    let (gltf, buffers, mut images) = gltf::import(path_for_gltf_model(name))?;

    drop(importing_gltf_span);

    {
        let _span = tracy_client::span!("Converting images");

        for image in &mut images {
            if image.format == gltf::image::Format::R8G8B8 {
                let dynamic_image = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(
                        image.width,
                        image.height,
                        std::mem::take(&mut image.pixels),
                    )
                    .unwrap(),
                );

                let rgba8 = dynamic_image.to_rgba8();

                image.format = gltf::image::Format::R8G8B8A8;
                image.pixels = rgba8.into_raw();
            }
        }
    }

    let node_tree = NodeTree::new(gltf.nodes());

    let loading_meshes_span = tracy_client::span!("Loading meshes");

    for (node, mesh) in gltf
        .nodes()
        .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
    {
        let transform = base_transform * node_tree.transform_of(node.index());

        for primitive in mesh.primitives() {
            let material = primitive.material();

            let draw_buffer_index = match (material.alpha_mode(), material.transmission().is_some())
            {
                (gltf::material::AlphaMode::Opaque, false) => 0,
                (gltf::material::AlphaMode::Mask, false) => 1,
                (gltf::material::AlphaMode::Opaque, true) => 2,
                (gltf::material::AlphaMode::Mask, true) => 3,
                (mode, _) => {
                    dbg!(mode);
                    0
                }
            };

            match draw_buffer_index {
                0 => max_draw_counts.opaque += 1,
                1 => max_draw_counts.alpha_clip += 1,
                2 => max_draw_counts.transmission += 1,
                _ => max_draw_counts.transmission_alpha_clip += 1,
            }

            // We handle texture transforms, but only scaling and only on the base colour texture.
            // This is the bare minimum to render the SheenCloth correctly.
            let uv_scaling = material
                .pbr_metallic_roughness()
                .base_color_texture()
                .and_then(|info| info.texture_transform())
                .map(|transform| Vec2::from(transform.scale()))
                .unwrap_or(Vec2::ONE);

            let material_id = material.index().unwrap_or(0) + model_buffers.materials.len();

            let reader = primitive.reader(|i| Some(&buffers[i.index()]));

            let read_indices = reader.read_indices().unwrap().into_u32();

            let num_existing_vertices = model_buffers.position.len();

            let first_index = model_buffers.index.len();

            model_buffers
                .index
                .extend(read_indices.map(|index| index + num_existing_vertices as u32));

            let first_instance = model_buffers.instances.len();

            model_buffers
                .position
                .extend(reader.read_positions().unwrap().map(Vec3::from));

            let num_current_vertices = model_buffers.position.len() - num_existing_vertices;

            model_buffers
                .material_id
                .extend(std::iter::repeat(material_id as u32).take(num_current_vertices));
            model_buffers
                .normal
                .extend(reader.read_normals().unwrap().map(Vec3::from));

            // Some test models (AttenuationTest) don't have UVs on some primitives.
            match reader.read_tex_coords(0) {
                Some(uvs) => {
                    model_buffers
                        .uv
                        .extend(uvs.into_f32().map(|uv| uv_scaling * Vec2::from(uv)));
                }
                None => {
                    model_buffers
                        .uv
                        .extend(std::iter::repeat(Vec2::ZERO).take(num_current_vertices));
                }
            }

            model_buffers.instances.push(Instance {
                transform: transform.pack(),
                primitive_id: model_buffers.primitives.len() as u32,
            });

            model_buffers
                .primitives
                .push(shared_structs::PrimitiveInfo {
                    packed_bounding_sphere: {
                        let bbox = primitive.bounding_box();
                        let min = Vec3::from(bbox.min);
                        let max = Vec3::from(bbox.max);
                        let center = (min + max) / 2.0;
                        let radius = min.distance(max) / 2.0;
                        center.extend(radius)
                    },
                    index_count: (model_buffers.index.len() - first_index) as u32,
                    first_index: first_index as u32,
                    first_instance: first_instance as u32,
                    draw_buffer_index,
                });
        }
    }

    drop(loading_meshes_span);

    let mut image_index_to_id = std::collections::HashMap::new();

    let loading_materials_span = tracy_client::span!("Loading materials");

    for (i, material) in gltf.materials().enumerate() {
        let mut load_optional_texture =
            |optional_texture_info: Option<gltf::texture::Texture>, name: &str, srgb| {
                match optional_texture_info {
                    Some(info) => {
                        let image_index = info.source().index();

                        let id = match image_index_to_id.entry((image_index, srgb)) {
                            std::collections::hash_map::Entry::Occupied(occupied) => {
                                println!("reusing image {} (srgb: {})", image_index, srgb);
                                *occupied.get()
                            }
                            std::collections::hash_map::Entry::Vacant(vacancy) => {
                                let image = &images[image_index];

                                let texture = load_texture_from_gltf(
                                    image,
                                    srgb,
                                    &format!("{} {}", name, i),
                                    init_resources,
                                    buffers_to_cleanup,
                                )?;

                                let id = image_manager.push_image(texture);

                                vacancy.insert(id);

                                id
                            }
                        };

                        Ok::<_, anyhow::Error>(id as i32)
                    }
                    None => Ok(-1),
                }
            };

        let pbr = material.pbr_metallic_roughness();

        model_buffers.materials.push(shared_structs::MaterialInfo {
            textures: shared_structs::Textures {
                diffuse: load_optional_texture(
                    pbr.base_color_texture().map(|info| info.texture()),
                    "diffuse",
                    true,
                )?,
                metallic_roughness: load_optional_texture(
                    pbr.metallic_roughness_texture().map(|info| info.texture()),
                    "metallic roughness",
                    false,
                )?,
                normal_map: load_optional_texture(
                    material.normal_texture().map(|info| info.texture()),
                    "normal map",
                    false,
                )?,
                emissive: load_optional_texture(
                    material.emissive_texture().map(|info| info.texture()),
                    "emissive",
                    true,
                )?,
                occlusion: load_optional_texture(
                    material.occlusion_texture().map(|info| info.texture()),
                    "occlusion",
                    true,
                )?,
                transmission: load_optional_texture(
                    material
                        .transmission()
                        .and_then(|transmission| transmission.transmission_texture())
                        .map(|info| info.texture()),
                    "transmission",
                    false,
                )?,
                thickness: load_optional_texture(
                    material
                        .volume()
                        .and_then(|volume| volume.thickness_texture())
                        .map(|info| info.texture()),
                    "volume",
                    false,
                )?,
                // todo: use a srgb/linear/dontcare enum for better texture re-use (textures that use alpha channels dont care about srgb).
                specular: load_optional_texture(
                    material
                        .specular()
                        .and_then(|specular| specular.specular_texture())
                        .map(|info| info.texture()),
                    "specular",
                    false,
                )?,
                specular_colour: load_optional_texture(
                    material
                        .specular()
                        .and_then(|specular| specular.specular_color_texture())
                        .map(|info| info.texture()),
                    "specular colour",
                    false,
                )?,
            },
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            alpha_clipping_cutoff: material.alpha_cutoff().unwrap_or(0.5),
            diffuse_factor: pbr.base_color_factor().into(),
            emissive_factor: material.emissive_factor().into(),
            normal_map_scale: material
                .normal_texture()
                .map(|normal_texture| normal_texture.scale())
                .unwrap_or_default(),
            occlusion_strength: material
                .occlusion_texture()
                .map(|occlusion| occlusion.strength())
                .unwrap_or(1.0),
            index_of_refraction: material.ior().unwrap_or(1.5),
            transmission_factor: material
                .transmission()
                .map(|transmission| transmission.transmission_factor())
                .unwrap_or(0.0),
            thickness_factor: material
                .volume()
                .map(|volume| volume.thickness_factor())
                .unwrap_or(0.0),
            attenuation_distance: material
                .volume()
                .map(|volume| volume.attenuation_distance() * base_transform.scale)
                .unwrap_or(f32::INFINITY),
            attenuation_colour: material
                .volume()
                .map(|volume| volume.attenuation_color())
                .unwrap_or([1.0; 3])
                .into(),
            specular_factor: material
                .specular()
                .map(|specular| specular.specular_factor())
                .unwrap_or(1.0),
            specular_colour_factor: material
                .specular()
                .map(|specular| specular.specular_color_factor())
                .unwrap_or([1.0; 3])
                .into(),
        });
    }

    drop(loading_materials_span);

    Ok(())
}

fn load_texture_from_gltf(
    image: &gltf::image::Data,
    srgb: bool,
    name: &str,
    init_resources: &mut ash_abstractions::InitResources,
    buffers_to_cleanup: &mut Vec<ash_abstractions::Buffer>,
) -> anyhow::Result<ash_abstractions::Image> {
    let format = match (image.format, srgb) {
        (gltf::image::Format::R8G8B8A8, true) => vk::Format::R8G8B8A8_SRGB,
        (gltf::image::Format::R8G8B8A8, false) => vk::Format::R8G8B8A8_UNORM,
        format => panic!("unsupported format: {:?}", format),
    };

    let mip_levels = mip_levels_for_size(image.width, image.height);

    let (image, staging_buffer) = ash_abstractions::load_image_from_bytes(
        &ash_abstractions::LoadImageDescriptor {
            bytes: &image.pixels,
            extent: vk::Extent3D {
                width: image.width,
                height: image.height,
                depth: 1,
            },
            view_ty: vk::ImageViewType::TYPE_2D,
            format,
            name,
            next_accesses: &[
                vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            ],
            next_layout: vk_sync::ImageLayout::Optimal,
            mip_levels,
        },
        init_resources,
    )?;

    buffers_to_cleanup.push(staging_buffer);

    Ok(image)
}

fn path_for_gltf_model(model: &str) -> PathBuf {
    let mut path = PathBuf::new();
    path.push("glTF-Sample-Models");
    path.push("2.0");
    path.push(model);
    path.push("glTF");
    path.push(model);
    path.set_extension("gltf");
    path
}

#[derive(Default)]
pub struct ImageManager {
    images: Vec<ash_abstractions::Image>,
    image_infos: Vec<vk::DescriptorImageInfo>,
}

impl ImageManager {
    pub fn push_image(&mut self, image: ash_abstractions::Image) -> u32 {
        let index = self.images.len() as u32;

        self.image_infos.push(
            *vk::DescriptorImageInfo::builder()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        );

        self.images.push(image);

        index
    }

    pub fn fill_with_dummy_images_up_to(&mut self, items: usize) {
        while self.image_infos.len() < items {
            self.image_infos.push(self.image_infos[0]);
        }
    }

    pub fn write_descriptor_set(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
    ) -> vk::WriteDescriptorSetBuilder {
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&self.image_infos)
    }

    pub fn cleanup(
        &self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> anyhow::Result<()> {
        for image in &self.images {
            image.cleanup(device, allocator)?;
        }

        Ok(())
    }
}

pub struct NodeTree {
    inner: Vec<(Similarity, usize)>,
}

impl NodeTree {
    fn new(nodes: gltf::iter::Nodes) -> Self {
        let mut inner = vec![(Similarity::IDENTITY, usize::max_value()); nodes.clone().count()];

        for node in nodes {
            let (translation, rotation, scale) = node.transform().decomposed();

            assert!((scale[0] - scale[1]).abs() <= std::f32::EPSILON);
            assert!((scale[0] - scale[2]).abs() <= std::f32::EPSILON);

            inner[node.index()].0 = Similarity {
                translation: translation.into(),
                rotation: Quat::from_array(rotation),
                scale: scale[0],
            };
            for child in node.children() {
                inner[child.index()].1 = node.index();
            }
        }

        Self { inner }
    }

    pub fn transform_of(&self, mut index: usize) -> Similarity {
        let mut transform_sum = Similarity::IDENTITY;

        while index != usize::max_value() {
            let (transform, parent_index) = self.inner[index];
            transform_sum = transform * transform_sum;
            index = parent_index;
        }

        transform_sum
    }
}

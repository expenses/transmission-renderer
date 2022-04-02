use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::vk;
use ash_reflect::FetchedDescriptorSetLayout;

pub struct DescriptorSetLayouts {
    pub main: FetchedDescriptorSetLayout,
    pub instance_buffer: FetchedDescriptorSetLayout,
    pub single_sampled_image: FetchedDescriptorSetLayout,
    pub frustum_culling: FetchedDescriptorSetLayout,
    pub lights: FetchedDescriptorSetLayout,
    pub cluster_data: FetchedDescriptorSetLayout,
    pub acceleration_structure_debugging: FetchedDescriptorSetLayout,
    pub depth_buffer: FetchedDescriptorSetLayout,
    pub sun_shadow_buffer: FetchedDescriptorSetLayout,
    pub packed_shadow_bitmasks: FetchedDescriptorSetLayout,
    pub tile_classification: FetchedDescriptorSetLayout,
    pub layouts: ash_reflect::BuiltDescriptorSetLayouts,
}

impl DescriptorSetLayouts {
    pub fn from_reflected_layouts(
        device: &ash::Device,
        debug_utils_loader: &DebugUtilsLoader,
        layouts: ash_reflect::BuiltDescriptorSetLayouts,
    ) -> anyhow::Result<Self> {
        log::debug!("layouts: {:#?}", &layouts);

        let create_and_name = |name, set_id| {
            let layout = layouts.layout_for_shader(name, set_id)?;

            ash_abstractions::set_object_name(
                device,
                debug_utils_loader,
                *layout,
                &format!("{} set {}", name, set_id),
            )?;

            Ok::<_, anyhow::Error>(layout)
        };

        Ok(Self {
            main: create_and_name("fragment::opaque", 0)?,
            instance_buffer: create_and_name("vertex::instanced", 1)?,
            single_sampled_image: create_and_name("fragment::tonemap", 1)?,
            frustum_culling: create_and_name("frustum_culling", 0)?,
            lights: create_and_name("fragment::opaque", 2)?,
            cluster_data: create_and_name("write_cluster_data", 1)?,
            acceleration_structure_debugging: create_and_name(
                "debugging::acceleration_structure_debugging",
                1,
            )?,
            depth_buffer: create_and_name("ray_trace_sun_shadow", 1)?,
            sun_shadow_buffer: create_and_name("reconstruct_shadow_buffer", 0)?,
            packed_shadow_bitmasks: create_and_name("ray_trace_sun_shadow", 2)?,
            tile_classification: create_and_name("tile_classification", 0)?,
            layouts,
        })
    }
}

pub struct DescriptorSets {
    pub main: vk::DescriptorSet,
    pub instance_buffer: vk::DescriptorSet,
    pub hdr_framebuffer: vk::DescriptorSet,
    pub opaque_sampled_hdr_framebuffer: vk::DescriptorSet,
    pub frustum_culling: vk::DescriptorSet,
    pub lights: vk::DescriptorSet,
    pub cluster_data: vk::DescriptorSet,
    pub acceleration_structure_debugging: vk::DescriptorSet,
    pub sun_shadow_buffer: vk::DescriptorSet,
    pub depth_buffer_a: vk::DescriptorSet,
    pub depth_buffer_b: vk::DescriptorSet,
    pub packed_shadow_bitmasks: vk::DescriptorSet,
    pub tile_classification_a: vk::DescriptorSet,
    pub tile_classification_b: vk::DescriptorSet,
    _descriptor_pool: vk::DescriptorPool,
}

impl DescriptorSets {
    pub fn allocate(device: &ash::Device, layouts: &DescriptorSetLayouts) -> anyhow::Result<Self> {
        let set_layouts = [
            layouts.main,
            layouts.instance_buffer,
            layouts.single_sampled_image,
            layouts.single_sampled_image,
            layouts.frustum_culling,
            layouts.lights,
            layouts.cluster_data,
            layouts.acceleration_structure_debugging,
            layouts.sun_shadow_buffer,
            layouts.depth_buffer,
            layouts.depth_buffer,
            layouts.packed_shadow_bitmasks,
            layouts.tile_classification,
            layouts.tile_classification,
        ];

        let pool_sizes = layouts.layouts.get_pool_sizes(&set_layouts);

        dbg!(&pool_sizes);

        let pool_sizes = pool_sizes.as_vec();

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .pool_sizes(&pool_sizes)
                    .max_sets(set_layouts.len() as u32),
                None,
            )
        }?;

        let set_layouts = set_layouts.map(|fetched_set_layout| *fetched_set_layout);

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .set_layouts(&set_layouts)
                    .descriptor_pool(descriptor_pool),
            )
        }?;

        Ok(Self {
            main: descriptor_sets[0],
            instance_buffer: descriptor_sets[1],
            hdr_framebuffer: descriptor_sets[2],
            opaque_sampled_hdr_framebuffer: descriptor_sets[3],
            frustum_culling: descriptor_sets[4],
            lights: descriptor_sets[5],
            cluster_data: descriptor_sets[6],
            acceleration_structure_debugging: descriptor_sets[7],
            sun_shadow_buffer: descriptor_sets[8],
            depth_buffer_a: descriptor_sets[9],
            depth_buffer_b: descriptor_sets[10],
            packed_shadow_bitmasks: descriptor_sets[11],
            tile_classification_a: descriptor_sets[12],
            tile_classification_b: descriptor_sets[13],
            _descriptor_pool: descriptor_pool,
        })
    }

    pub fn update_framebuffers(
        &self,
        device: &ash::Device,
        hdr_framebuffer: &ash_abstractions::Image,
        opaque_sampled_hdr_framebuffer: &ash_abstractions::Image,
        sun_shadow_buffer: &ash_abstractions::Image,
        packed_shadow_bitmasks_buffer: &ash_abstractions::Buffer,
    ) {
        unsafe {
            device.update_descriptor_sets(
                &[
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(self.hdr_framebuffer)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[*vk::DescriptorImageInfo::builder()
                            .image_view(hdr_framebuffer.view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(self.opaque_sampled_hdr_framebuffer)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[*vk::DescriptorImageInfo::builder()
                            .image_view(opaque_sampled_hdr_framebuffer.view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(self.acceleration_structure_debugging)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[*vk::DescriptorImageInfo::builder()
                            .image_view(hdr_framebuffer.view)
                            .image_layout(vk::ImageLayout::GENERAL)]),
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(self.sun_shadow_buffer)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[*vk::DescriptorImageInfo::builder()
                            .image_view(sun_shadow_buffer.view)
                            .image_layout(vk::ImageLayout::GENERAL)]),
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(self.packed_shadow_bitmasks)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: packed_shadow_bitmasks_buffer.buffer,
                            range: vk::WHOLE_SIZE,
                            offset: 0,
                        }]),
                ],
                &[],
            );
        }
    }
}

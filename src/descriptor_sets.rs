use crate::MAX_IMAGES;
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
    pub g_buffer: FetchedDescriptorSetLayout,
    pub sun_shadow_buffer: FetchedDescriptorSetLayout,
    layouts: ash_reflect::DescriptorSetLayouts,
}

impl DescriptorSetLayouts {
    pub fn from_reflected_layouts(
        device: &ash::Device,
        layouts: ash_reflect::DescriptorSetLayouts,
    ) -> anyhow::Result<Self> {
        let settings = ash_reflect::Settings {
            max_unbounded_descriptors: MAX_IMAGES,
            enable_partially_bound_unbounded_descriptors: true,
        };

        Ok(Self {
            main: layouts.layout_for_shader(device, "fragment::opaque", 0, settings)?,
            instance_buffer: layouts.layout_for_shader(device, "vertex::instanced", 1, settings)?,
            single_sampled_image: layouts.layout_for_shader(
                device,
                "fragment::tonemap",
                1,
                settings,
            )?,
            frustum_culling: layouts.layout_for_shader(device, "frustum_culling", 0, settings)?,
            lights: layouts.layout_for_shader(device, "fragment::opaque", 2, settings)?,
            cluster_data: layouts.layout_for_shader(device, "write_cluster_data", 1, settings)?,
            acceleration_structure_debugging: layouts.layout_for_shader(
                device,
                "debugging::acceleration_structure_debugging",
                1,
                settings,
            )?,
            g_buffer: layouts.layout_for_shader(device, "ray_trace_sun_shadow", 1, settings)?,
            sun_shadow_buffer: layouts.layout_for_shader(
                device,
                "ray_trace_sun_shadow",
                2,
                settings,
            )?,
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
    pub g_buffer: vk::DescriptorSet,
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
            layouts.g_buffer,
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
            g_buffer: descriptor_sets[9],
            _descriptor_pool: descriptor_pool,
        })
    }

    pub fn update_framebuffers(
        &self,
        device: &ash::Device,
        hdr_framebuffer: &ash_abstractions::Image,
        opaque_sampled_hdr_framebuffer: &ash_abstractions::Image,
        sun_shadow_buffer: &ash_abstractions::Image,
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
                ],
                &[],
            );
        }
    }
}

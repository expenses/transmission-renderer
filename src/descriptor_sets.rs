use crate::MAX_IMAGES;
use ash::vk;

pub struct DescriptorSetLayouts {
    pub main: vk::DescriptorSetLayout,
    pub instance_buffer: vk::DescriptorSetLayout,
    pub single_sampled_image: vk::DescriptorSetLayout,
    pub frustum_culling: vk::DescriptorSetLayout,
    pub lights: vk::DescriptorSetLayout,
    pub cluster_data: vk::DescriptorSetLayout,
    pub acceleration_structure_debugging: vk::DescriptorSetLayout,
    pub g_buffer: vk::DescriptorSetLayout,
    pub sun_shadow_buffer: vk::DescriptorSetLayout,
}

impl DescriptorSetLayouts {
    pub fn from_reflected_layouts(built: &ash_reflect::BuiltDescriptorSetLayouts) -> Self {
        Self {
            main: built.layout_for_shader("fragment::opaque", 0),
            instance_buffer: built.layout_for_shader("vertex::instanced", 1),
            single_sampled_image: built.layout_for_shader("fragment::tonemap", 1),
            frustum_culling: built.layout_for_shader("frustum_culling", 0),
            lights: built.layout_for_shader("fragment::opaque", 2),
            cluster_data: built.layout_for_shader("write_cluster_data", 1),
            acceleration_structure_debugging: built.layout_for_shader("debugging::acceleration_structure_debugging", 1),
            g_buffer: built.layout_for_shader("ray_trace_sun_shadow", 1),
            sun_shadow_buffer: built.layout_for_shader("ray_trace_sun_shadow", 2),
        }
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
    pub fn allocate(device: &ash::Device, layouts: &DescriptorSetLayouts, pool_sizes: &ash_reflect::PoolSizes) -> anyhow::Result<Self> {
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

        let mut pool_sizes = pool_sizes.as_vec();

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .pool_sizes(&pool_sizes)
                    .max_sets(set_layouts.len() as u32),
                None,
            )
        }?;

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

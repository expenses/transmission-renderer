use crate::MAX_IMAGES;
use ash::vk;

pub struct DescriptorSetLayouts {
    pub main: vk::DescriptorSetLayout,
    pub instance_buffer: vk::DescriptorSetLayout,
    pub hdr_framebuffer: vk::DescriptorSetLayout,
    pub frustum_culling: vk::DescriptorSetLayout,
}

impl DescriptorSetLayouts {
    pub fn new(device: &ash::Device) -> anyhow::Result<Self> {
        Ok(Self {
            main: unsafe {
                device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(MAX_IMAGES)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(3)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(4)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(5)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    ]),
                    None,
                )?
            },
            hdr_framebuffer: unsafe {
                device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    ]),
                    None,
                )?
            },
            frustum_culling: unsafe {
                device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(3)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(4)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(5)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(6)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    ]),
                    None,
                )?
            },
            instance_buffer: unsafe {
                device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE),
                    ]),
                    None,
                )?
            }
        })
    }
}

pub struct DescriptorSets {
    pub main: vk::DescriptorSet,
    pub instance_buffer: vk::DescriptorSet,
    pub hdr_framebuffer: vk::DescriptorSet,
    pub opaque_sampled_hdr_framebuffer: vk::DescriptorSet,
    pub frustum_culling: vk::DescriptorSet,
    _descriptor_pool: vk::DescriptorPool,
}

impl DescriptorSets {
    pub fn allocate(device: &ash::Device, layouts: &DescriptorSetLayouts) -> anyhow::Result<Self> {
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .pool_sizes(&[
                        *vk::DescriptorPoolSize::builder()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(3 + 8),
                        *vk::DescriptorPoolSize::builder()
                            .ty(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(2),
                        *vk::DescriptorPoolSize::builder()
                            .ty(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(MAX_IMAGES),
                        *vk::DescriptorPoolSize::builder()
                            .ty(vk::DescriptorType::SAMPLER)
                            .descriptor_count(2),
                    ])
                    .max_sets(5),
                None,
            )
        }?;

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .set_layouts(&[
                        layouts.main,
                        layouts.instance_buffer,
                        layouts.hdr_framebuffer,
                        layouts.hdr_framebuffer,
                        layouts.frustum_culling,
                    ])
                    .descriptor_pool(descriptor_pool),
            )
        }?;

        Ok(Self {
            main: descriptor_sets[0],
            instance_buffer: descriptor_sets[1],
            hdr_framebuffer: descriptor_sets[2],
            opaque_sampled_hdr_framebuffer: descriptor_sets[3],
            frustum_culling: descriptor_sets[4],
            _descriptor_pool: descriptor_pool,
        })
    }

    pub fn update_framebuffers(
        &self,
        device: &ash::Device,
        hdr_framebuffer: &ash_abstractions::Image,
        opaque_sampled_hdr_framebuffer: &ash_abstractions::Image,
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
                ],
                &[],
            );
        }
    }
}

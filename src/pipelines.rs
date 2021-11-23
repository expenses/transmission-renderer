use ash::vk;
use std::ffi::CString;
use crate::render_passes::RenderPasses;
use crate::descriptor_set_layouts::DescriptorSetLayouts;
use glam::{Vec2, Vec3};

pub struct Pipelines {
    pub normal: vk::Pipeline,
    pub depth_pre_pass: vk::Pipeline,
    pub depth_pre_pass_alpha_clip: vk::Pipeline,
    pub depth_pre_pass_transmissive: vk::Pipeline,
    pub depth_pre_pass_transmissive_alpha_clip: vk::Pipeline,
    pub transmission: vk::Pipeline,
    pub tonemap: vk::Pipeline,
    pub frustum_culling: vk::Pipeline,
    pub demultiplex_draws: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub tonemap_pipeline_layout: vk::PipelineLayout,
    pub transmission_pipeline_layout: vk::PipelineLayout,
    pub frustum_culling_pipeline_layout: vk::PipelineLayout,
}

impl Pipelines {
    pub fn new(
        device: &ash::Device,
        render_passes: &RenderPasses,
        descriptor_set_layouts: &DescriptorSetLayouts,
        pipeline_cache: vk::PipelineCache,
    ) -> anyhow::Result<Self> {
        let _span = tracy_client::span!("Pipelines::new");

        let vertex_instanced_entry_point = CString::new("vertex_instanced")?;
        let vertex_instanced_with_scale_entry_point = CString::new("vertex_instanced_with_scale")?;
        let fragment_entry_point = CString::new("fragment")?;
        let vertex_depth_pre_pass_entry_point = CString::new("depth_pre_pass_instanced")?;
        let vertex_depth_pre_pass_alpha_clip_entry_point =
            CString::new("depth_pre_pass_vertex_alpha_clip")?;
        let fragment_depth_pre_pass_alpha_clip_entry_point =
            CString::new("depth_pre_pass_alpha_clip")?;
        let fragment_transmission_entry_point = CString::new("fragment_transmission")?;
        let fullscreen_tri_entry_point = CString::new("fullscreen_tri")?;
        let fragment_tonemap_entry_point = CString::new("fragment_tonemap")?;

        let frustum_culling_entry_point = CString::new("frustum_culling")?;
        let demultiplex_draws_entry_point = CString::new("demultiplex_draws")?;

        let module = ash_abstractions::load_shader_module(include_bytes!("../shader.spv"), device)?;

        let fragment_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_entry_point);

        let fragment_transmission_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_transmission_entry_point);

        let vertex_instanced_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_instanced_entry_point);

        let vertex_instanced_with_scale_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_instanced_with_scale_entry_point);

        let vertex_depth_pre_pass_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_depth_pre_pass_entry_point);

        let vertex_depth_pre_pass_alpha_clip_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_depth_pre_pass_alpha_clip_entry_point);

        let fragment_depth_pre_pass_alpha_clip_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_depth_pre_pass_alpha_clip_entry_point);

        let fullscreen_tri_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&fullscreen_tri_entry_point);

        let fragment_tonemap_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_tonemap_entry_point);

        let frustum_culling_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&frustum_culling_entry_point);

        let demultiplex_draws_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&demultiplex_draws_entry_point);

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layouts.main])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
                None,
            )
        }?;

        let transmission_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.main,
                        descriptor_set_layouts.hdr_framebuffer,
                    ])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
                None,
            )
        }?;

        let frustum_culling_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layouts.frustum_culling])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .size(std::mem::size_of::<shared_structs::CullingPushConstants>() as u32)]),
                None,
            )
        }?;

        let tonemap_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.main,
                        descriptor_set_layouts.hdr_framebuffer,
                    ])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .size(
                            std::mem::size_of::<colstodian::tonemap::BakedLottesTonemapperParams>()
                                as u32,
                        )]),
                None,
            )
        }?;

        let full_vertex_attributes = ash_abstractions::create_vertex_attribute_descriptions(&[
            &[ash_abstractions::VertexAttribute::Vec3],
            &[ash_abstractions::VertexAttribute::Vec3],
            &[ash_abstractions::VertexAttribute::Vec2],
            &[ash_abstractions::VertexAttribute::Uint],
        ]);

        let full_vertex_bindings = [
            *vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(std::mem::size_of::<Vec3>() as u32),
            *vk::VertexInputBindingDescription::builder()
                .binding(1)
                .stride(std::mem::size_of::<Vec3>() as u32),
            *vk::VertexInputBindingDescription::builder()
                .binding(2)
                .stride(std::mem::size_of::<Vec2>() as u32),
            *vk::VertexInputBindingDescription::builder()
                .binding(3)
                .stride(std::mem::size_of::<u32>() as u32),
        ];

        let normal_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: false,
                depth_compare_op: vk::CompareOp::EQUAL,
            }),
            vertex_attributes: &full_vertex_attributes,
            vertex_bindings: &full_vertex_bindings,
            colour_attachments: &[
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false),
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false),
            ],
        };

        let transmission_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::EQUAL,
            }),
            vertex_attributes: &full_vertex_attributes,
            vertex_bindings: &full_vertex_bindings,
            colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)],
        };

        let depth_pre_pass_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::LESS,
            }),
            vertex_attributes: &ash_abstractions::create_vertex_attribute_descriptions(&[
                &[ash_abstractions::VertexAttribute::Vec3],
                &[],
                &[],
                &[],
            ]),
            vertex_bindings: &[*vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(std::mem::size_of::<Vec3>() as u32)],
            colour_attachments: &[],
        };

        let depth_pre_pass_alpha_clip_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::LESS,
            }),
            vertex_attributes: &ash_abstractions::create_vertex_attribute_descriptions(&[
                &[ash_abstractions::VertexAttribute::Vec3],
                &[],
                &[ash_abstractions::VertexAttribute::Vec2],
                &[ash_abstractions::VertexAttribute::Uint],
            ]),
            vertex_bindings: &[
                *vk::VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<Vec3>() as u32),
                *vk::VertexInputBindingDescription::builder()
                    .binding(2)
                    .stride(std::mem::size_of::<Vec2>() as u32),
                *vk::VertexInputBindingDescription::builder()
                    .binding(3)
                    .stride(std::mem::size_of::<u32>() as u32),
            ],
            colour_attachments: &[],
        };

        let tonemap_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::NONE,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: None,
            vertex_attributes: &[],
            vertex_bindings: &[],
            colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)],
        };

        let normal_baked = normal_pipeline_desc.as_baked();
        let depth_pre_pass_baked = depth_pre_pass_desc.as_baked();
        let depth_pre_pass_alpha_clip_baked = depth_pre_pass_alpha_clip_desc.as_baked();
        let transmission_baked = transmission_pipeline_desc.as_baked();
        let tonemap_pipeline_baked = tonemap_pipeline_desc.as_baked();

        let stages = &[*vertex_instanced_stage, *fragment_stage];

        let normal_pipeline_desc =
            normal_baked.as_pipeline_create_info(stages, pipeline_layout, render_passes.draw, 1);

        let depth_pre_pass_stage = &[*vertex_depth_pre_pass_stage];

        let depth_pre_pass_desc = depth_pre_pass_baked.as_pipeline_create_info(
            depth_pre_pass_stage,
            pipeline_layout,
            render_passes.draw,
            0,
        );

        let depth_pre_pass_alpha_clip_stages = &[
            *vertex_depth_pre_pass_alpha_clip_stage,
            *fragment_depth_pre_pass_alpha_clip_stage,
        ];

        let depth_pre_pass_alpha_clip_desc = depth_pre_pass_alpha_clip_baked
            .as_pipeline_create_info(
                depth_pre_pass_alpha_clip_stages,
                pipeline_layout,
                render_passes.draw,
                0,
            );

        let transmission_stages = &[
            *vertex_instanced_with_scale_stage,
            *fragment_transmission_stage,
        ];

        let transmission_pipeline_desc = transmission_baked.as_pipeline_create_info(
            transmission_stages,
            transmission_pipeline_layout,
            render_passes.transmission,
            0,
        );

        let tonemap_stages = &[*fullscreen_tri_stage, *fragment_tonemap_stage];

        let tonemap_pipeline_desc = tonemap_pipeline_baked.as_pipeline_create_info(
            tonemap_stages,
            tonemap_pipeline_layout,
            render_passes.tonemap,
            0,
        );

        let depth_pre_pass_transmissive_desc = depth_pre_pass_baked.as_pipeline_create_info(
            depth_pre_pass_stage,
            pipeline_layout,
            render_passes.draw,
            2,
        );

        let depth_pre_pass_transmissive_alpha_clip_desc = depth_pre_pass_alpha_clip_baked
            .as_pipeline_create_info(
                depth_pre_pass_alpha_clip_stages,
                pipeline_layout,
                render_passes.draw,
                2,
            );

        let frustum_culling_desc = vk::ComputePipelineCreateInfo::builder()
            .stage(*frustum_culling_stage)
            .layout(frustum_culling_pipeline_layout);

        let demultiplex_draws_desc = vk::ComputePipelineCreateInfo::builder()
            .stage(*demultiplex_draws_stage)
            .layout(frustum_culling_pipeline_layout);

        let pipelines = unsafe {
            device.create_graphics_pipelines(
                pipeline_cache,
                &[
                    *normal_pipeline_desc,
                    *depth_pre_pass_desc,
                    *depth_pre_pass_alpha_clip_desc,
                    *depth_pre_pass_transmissive_desc,
                    *depth_pre_pass_transmissive_alpha_clip_desc,
                    *transmission_pipeline_desc,
                    *tonemap_pipeline_desc,
                ],
                None,
            )
        }
        .map_err(|(_, err)| err)?;

        let compute_pipelines = unsafe {
            device.create_compute_pipelines(
                pipeline_cache,
                &[*frustum_culling_desc, *demultiplex_draws_desc],
                None,
            )
        }
        .map_err(|(_, err)| err)?;

        Ok(Self {
            normal: pipelines[0],
            depth_pre_pass: pipelines[1],
            depth_pre_pass_alpha_clip: pipelines[2],
            depth_pre_pass_transmissive: pipelines[3],
            depth_pre_pass_transmissive_alpha_clip: pipelines[4],
            transmission: pipelines[5],
            tonemap: pipelines[6],
            frustum_culling: compute_pipelines[0],
            demultiplex_draws: compute_pipelines[1],
            pipeline_layout,
            tonemap_pipeline_layout,
            transmission_pipeline_layout,
            frustum_culling_pipeline_layout,
        })
    }
}

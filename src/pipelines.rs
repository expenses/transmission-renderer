use crate::descriptor_sets::DescriptorSetLayouts;
use crate::render_passes::RenderPasses;
use ash::vk;
use c_str_macro::c_str;
use glam::{Vec2, Vec3};
use std::path::{Path, PathBuf};

fn read_shader(parent: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let mut path = parent.join(name);
    path.set_extension("spv");
    Ok(std::fs::read(&path)?)
}

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
    pub assign_lights_to_froxels: vk::Pipeline,
    pub write_froxel_data: vk::Pipeline,
    pub acceleration_structure_debugging: Option<vk::Pipeline>,
    pub pipeline_layout: vk::PipelineLayout,
    pub tonemap_pipeline_layout: vk::PipelineLayout,
    pub transmission_pipeline_layout: vk::PipelineLayout,
    pub frustum_culling_pipeline_layout: vk::PipelineLayout,
    pub acceleration_structure_debugging_layout: vk::PipelineLayout,
    pub lights_pipeline_layout: vk::PipelineLayout,
    pub write_froxel_data_pipeline_layout: vk::PipelineLayout,
}

impl Pipelines {
    pub fn new(
        device: &ash::Device,
        render_passes: &RenderPasses,
        descriptor_set_layouts: &DescriptorSetLayouts,
        pipeline_cache: vk::PipelineCache,
        enable_ray_tracing: bool,
    ) -> anyhow::Result<Self> {
        let _span = tracy_client::span!("Pipelines::new");

        let normal = &PathBuf::from("compiled-shaders/normal");
        let ray_tracing = &PathBuf::from("compiled-shaders/ray-tracing");

        let maybe_ray_tracing = if enable_ray_tracing {
            ray_tracing
        } else {
            normal
        };

        let fragment_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(maybe_ray_tracing, "fragment")?,
            vk::ShaderStageFlags::FRAGMENT,
            device,
            c_str!("fragment"),
        )?;

        let fragment_transmission_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(maybe_ray_tracing, "fragment_transmission")?,
            vk::ShaderStageFlags::FRAGMENT,
            device,
            c_str!("fragment_transmission"),
        )?;

        let vertex_instanced_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "vertex_instanced")?,
            vk::ShaderStageFlags::VERTEX,
            device,
            c_str!("vertex_instanced"),
        )?;

        let vertex_instanced_with_scale_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "vertex_instanced_with_scale")?,
            vk::ShaderStageFlags::VERTEX,
            device,
            c_str!("vertex_instanced_with_scale"),
        )?;

        let vertex_depth_pre_pass_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "depth_pre_pass_instanced")?,
            vk::ShaderStageFlags::VERTEX,
            device,
            c_str!("depth_pre_pass_instanced"),
        )?;

        let vertex_depth_pre_pass_alpha_clip_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "depth_pre_pass_vertex_alpha_clip")?,
            vk::ShaderStageFlags::VERTEX,
            device,
            c_str!("depth_pre_pass_vertex_alpha_clip"),
        )?;

        let fragment_depth_pre_pass_alpha_clip_stage =
            ash_abstractions::load_shader_module_as_stage(
                &read_shader(normal, "depth_pre_pass_alpha_clip")?,
                vk::ShaderStageFlags::FRAGMENT,
                device,
                c_str!("depth_pre_pass_alpha_clip"),
            )?;

        let fullscreen_tri_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "fullscreen_tri")?,
            vk::ShaderStageFlags::VERTEX,
            device,
            c_str!("fullscreen_tri"),
        )?;

        let fragment_tonemap_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "fragment_tonemap")?,
            vk::ShaderStageFlags::FRAGMENT,
            device,
            c_str!("fragment_tonemap"),
        )?;

        let frustum_culling_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "frustum_culling")?,
            vk::ShaderStageFlags::COMPUTE,
            device,
            c_str!("frustum_culling"),
        )?;

        let demultiplex_draws_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "demultiplex_draws")?,
            vk::ShaderStageFlags::COMPUTE,
            device,
            c_str!("demultiplex_draws"),
        )?;

        let assign_lights_to_froxels_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "assign_lights_to_froxels")?,
            vk::ShaderStageFlags::COMPUTE,
            device,
            c_str!("assign_lights_to_froxels"),
        )?;

        let write_froxel_data_stage = ash_abstractions::load_shader_module_as_stage(
            &read_shader(normal, "write_froxel_data")?,
            vk::ShaderStageFlags::COMPUTE,
            device,
            c_str!("write_froxel_data"),
        )?;

        let acceleration_structure_debugging_stage = if enable_ray_tracing {
            Some(ash_abstractions::load_shader_module_as_stage(
                &read_shader(ray_tracing, "acceleration_structure_debugging")?,
                vk::ShaderStageFlags::COMPUTE,
                device,
                c_str!("acceleration_structure_debugging"),
            )?)
        } else {
            None
        };

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.main,
                        descriptor_set_layouts.instance_buffer,
                        descriptor_set_layouts.lights,
                    ])
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
                        descriptor_set_layouts.instance_buffer,
                        descriptor_set_layouts.lights,
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
                    .set_layouts(&[
                        descriptor_set_layouts.frustum_culling,
                        descriptor_set_layouts.instance_buffer,
                    ])
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

        let acceleration_structure_debugging_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layouts.acceleration_structure_debugging])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
                None,
            )
        }?;

        let write_froxel_data_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.froxel_data,
                    ])
                    .push_constant_ranges(&[]),
                None,
            )
        }?;

        let lights_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.lights,
                        descriptor_set_layouts.froxel_data,
                    ])
                    .push_constant_ranges(&[]),
                None,
            )
        }?;

        let full_vertex_attributes = ash_abstractions::create_vertex_attribute_descriptions(&[
            &[ash_abstractions::VertexAttribute::Vec3],
            &[ash_abstractions::VertexAttribute::Vec3],
            &[ash_abstractions::VertexAttribute::Vec2],
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
            ]),
            vertex_bindings: &[
                *vk::VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<Vec3>() as u32),
                *vk::VertexInputBindingDescription::builder()
                    .binding(2)
                    .stride(std::mem::size_of::<Vec2>() as u32),
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

        let mut compute_pipeline_stages = vec![
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*frustum_culling_stage)
                .layout(frustum_culling_pipeline_layout),
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*demultiplex_draws_stage)
                .layout(frustum_culling_pipeline_layout),
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*assign_lights_to_froxels_stage)
                .layout(lights_pipeline_layout),
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*write_froxel_data_stage)
                .layout(write_froxel_data_pipeline_layout)
        ];

        if let Some(stage) = acceleration_structure_debugging_stage.as_ref() {
            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(**stage)
                    .layout(acceleration_structure_debugging_layout),
            );
        }

        let compute_pipelines = unsafe {
            device.create_compute_pipelines(pipeline_cache, &compute_pipeline_stages, None)
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
            assign_lights_to_froxels: compute_pipelines[2],
            write_froxel_data: compute_pipelines[3],
            acceleration_structure_debugging: if acceleration_structure_debugging_stage.is_some() {
                Some(compute_pipelines[4])
            } else {
                None
            },
            pipeline_layout,
            tonemap_pipeline_layout,
            transmission_pipeline_layout,
            frustum_culling_pipeline_layout,
            acceleration_structure_debugging_layout,
            lights_pipeline_layout,
            write_froxel_data_pipeline_layout,
        })
    }
}

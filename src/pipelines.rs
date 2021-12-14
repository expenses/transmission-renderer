use crate::descriptor_sets::DescriptorSetLayouts;
use crate::render_passes::RenderPasses;
use ash::vk;
use glam::{Vec2, Vec3};
use std::path::{Path, PathBuf};

fn read_shader(parent: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let mut path = parent.join(name);
    path.set_extension("spv");
    Ok(std::fs::read(&path)?)
}

pub struct Pipelines {
    pub(crate) normal: vk::Pipeline,
    pub(crate) depth_pre_pass: vk::Pipeline,
    pub(crate) depth_pre_pass_alpha_clip: vk::Pipeline,
    pub(crate) depth_pre_pass_transmissive: vk::Pipeline,
    pub(crate) depth_pre_pass_transmissive_alpha_clip: vk::Pipeline,
    pub(crate) transmission: vk::Pipeline,
    pub(crate) tonemap: vk::Pipeline,
    pub(crate) frustum_culling: vk::Pipeline,
    pub(crate) demultiplex_draws: vk::Pipeline,
    pub(crate) assign_lights_to_clusters: vk::Pipeline,
    pub(crate) write_cluster_data: vk::Pipeline,
    pub(crate) cluster_debugging: vk::Pipeline,
    pub(crate) defer_opaque: vk::Pipeline,
    pub(crate) defer_alpha_clip: vk::Pipeline,
    pub(crate) acceleration_structure_debugging: Option<vk::Pipeline>,
    pub(crate) sun_shadow: Option<vk::Pipeline>,
    pub(crate) depth_pre_pass_pipeline_layout: vk::PipelineLayout,
    pub(crate) draw_pipeline_layout: vk::PipelineLayout,
    pub(crate) tonemap_pipeline_layout: vk::PipelineLayout,
    pub(crate) transmission_pipeline_layout: vk::PipelineLayout,
    pub(crate) frustum_culling_pipeline_layout: vk::PipelineLayout,
    pub(crate) acceleration_structure_debugging_layout: vk::PipelineLayout,
    pub(crate) lights_pipeline_layout: vk::PipelineLayout,
    pub(crate) write_cluster_data_pipeline_layout: vk::PipelineLayout,
    pub(crate) cluster_debugging_pipeline_layout: vk::PipelineLayout,
}

impl Pipelines {
    pub fn new(
        device: &ash::Device,
        render_passes: &RenderPasses,
        pipeline_cache: vk::PipelineCache,
        enable_ray_tracing: bool,
    ) -> anyhow::Result<(Self, DescriptorSetLayouts, ash_reflect::PoolSizes)> {
        let _span = tracy_client::span!("Pipelines::new");

        let normal = &PathBuf::from("compiled-shaders/normal");
        let ray_tracing = &PathBuf::from("compiled-shaders/ray-tracing");

        let maybe_ray_tracing = if enable_ray_tracing {
            ray_tracing
        } else {
            normal
        };

        let mut layouts = ash_reflect::DescriptorSetLayouts::default();

        let fragment_stage = layouts.passthrough(ash_reflect::ShaderModule::new(device, &read_shader(maybe_ray_tracing, "fragment_opaque")?)?);

        let fragment_transmission_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(maybe_ray_tracing, "fragment_transmission")?,
        )?);

        let vertex_instanced_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "vertex_instanced")?,
        )?);

        let vertex_instanced_with_scale_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "vertex_instanced_with_scale")?,
        )?);

        let vertex_depth_pre_pass_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "depth_pre_pass_vertex")?,
        )?);

        let vertex_depth_pre_pass_alpha_clip_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "depth_pre_pass_vertex_alpha_clip")?,
        )?);

        let fragment_depth_pre_pass_alpha_clip_stage =
        layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
                &read_shader(normal, "depth_pre_pass_fragment_alpha_clip")?,
            )?);

        let fullscreen_tri_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "vertex_fullscreen_tri")?,
        )?);

        let fragment_tonemap_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "fragment_tonemap")?,
        )?);

        let frustum_culling_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "frustum_culling")?,
        )?);

        let demultiplex_draws_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "demultiplex_draws")?,
        )?);

        let assign_lights_to_clusters_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "assign_lights_to_clusters")?,
        )?);

        let write_cluster_data_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "write_cluster_data")?,
        )?);

        let cluster_debugging_vs_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "debugging_cluster_debugging_vs")?,
        )?);

        let cluster_debugging_fs_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "debugging_cluster_debugging_fs")?,
        )?);

        let defer_opaque_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "deferred_defer_opaque")?,
        )?);

        let defer_alpha_clip_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "deferred_defer_alpha_clip")?,
        )?);

        let defer_vs_stage = layouts.passthrough(ash_reflect::ShaderModule::new(
            device,
            &read_shader(normal, "deferred_vs")?,
        )?);

        struct RayTracingStages {
            acceleration_structure_debugging: ash_reflect::ShaderModule,
            ray_trace_sun_shadow: ash_reflect::ShaderModule,
        }

        let ray_tracing_stages = if enable_ray_tracing {
            Some(RayTracingStages {
                acceleration_structure_debugging: ash_reflect::ShaderModule::new(
                    device,
                    &read_shader(ray_tracing, "debugging_acceleration_structure_debugging")?,
                )?,
                ray_trace_sun_shadow: ash_reflect::ShaderModule::new(
                    device,
                    &read_shader(ray_tracing, "ray_trace_sun_shadow")?,
                )?
            })
        } else {
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(
                &read_shader(ray_tracing, "debugging_acceleration_structure_debugging")?
            )?);
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(
                &read_shader(ray_tracing, "ray_trace_sun_shadow")?
            )?);

            None
        };

        dbg!(&layouts);
        let mut pool_sizes = layouts.get_pool_sizes(crate::MAX_IMAGES);
        pool_sizes.add(vk::DescriptorType::SAMPLED_IMAGE, 1);
        dbg!(&pool_sizes);
        let built_layouts = layouts.build(&device, crate::MAX_IMAGES, true)?;

        let descriptor_set_layouts = DescriptorSetLayouts::from_reflected_layouts(&built_layouts);

        let draw_pipeline_layout = unsafe {
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

        let depth_pre_pass_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.main,
                        descriptor_set_layouts.instance_buffer,
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
                        descriptor_set_layouts.single_sampled_image,
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
                        descriptor_set_layouts.single_sampled_image,
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
                    .set_layouts(&[
                        descriptor_set_layouts.main,
                        descriptor_set_layouts.acceleration_structure_debugging,
                        descriptor_set_layouts.instance_buffer,
                    ])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
                None,
            )
        }?;

        let write_cluster_data_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.main,
                        descriptor_set_layouts.cluster_data,
                    ])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .size(
                            std::mem::size_of::<shared_structs::WriteClusterDataPushConstants>()
                                as u32,
                        )]),
                None,
            )
        }?;

        let lights_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.lights,
                        descriptor_set_layouts.cluster_data,
                    ])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .size(
                            std::mem::size_of::<shared_structs::AssignLightsPushConstants>() as u32,
                        )]),
                None,
            )
        }?;

        let cluster_debugging_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layouts.cluster_data])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
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

        let defer_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::GREATER,
            }),
            vertex_attributes: &full_vertex_attributes,
            vertex_bindings: &full_vertex_bindings,
            colour_attachments: &[
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
                depth_compare_op: vk::CompareOp::GREATER,
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
                depth_compare_op: vk::CompareOp::GREATER,
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

        let cluster_debugging_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::NONE,
                topology: vk::PrimitiveTopology::LINE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::GREATER,
            }),
            vertex_attributes: &[],
            vertex_bindings: &[],
            colour_attachments: &[
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false),
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false),
            ],
        };

        let normal_baked = normal_pipeline_desc.as_baked();
        let depth_pre_pass_baked = depth_pre_pass_desc.as_baked();
        let depth_pre_pass_alpha_clip_baked = depth_pre_pass_alpha_clip_desc.as_baked();
        let transmission_baked = transmission_pipeline_desc.as_baked();
        let tonemap_pipeline_baked = tonemap_pipeline_desc.as_baked();
        let cluster_debugging_baked = cluster_debugging_pipeline_desc.as_baked();
        let defer_baked = defer_pipeline_desc.as_baked();

        let stages = &[*vertex_instanced_stage.as_stage_create_info(), *fragment_stage.as_stage_create_info()];

        let normal_pipeline_desc = normal_baked.as_pipeline_create_info(
            stages,
            draw_pipeline_layout,
            render_passes.draw,
            0,
        );

        let depth_pre_pass_stage = &[*vertex_depth_pre_pass_stage.as_stage_create_info()];

        let depth_pre_pass_desc = depth_pre_pass_baked.as_pipeline_create_info(
            depth_pre_pass_stage,
            depth_pre_pass_pipeline_layout,
            render_passes.depth_pre_pass,
            0,
        );

        let depth_pre_pass_alpha_clip_stages = &[
            *vertex_depth_pre_pass_alpha_clip_stage.as_stage_create_info(),
            *fragment_depth_pre_pass_alpha_clip_stage.as_stage_create_info(),
        ];

        let depth_pre_pass_alpha_clip_desc = depth_pre_pass_alpha_clip_baked
            .as_pipeline_create_info(
                depth_pre_pass_alpha_clip_stages,
                depth_pre_pass_pipeline_layout,
                render_passes.depth_pre_pass,
                0,
            );

        let transmission_stages = &[
            *vertex_instanced_with_scale_stage.as_stage_create_info(),
            *fragment_transmission_stage.as_stage_create_info(),
        ];

        let transmission_pipeline_desc = transmission_baked.as_pipeline_create_info(
            transmission_stages,
            transmission_pipeline_layout,
            render_passes.transmission,
            1,
        );

        let tonemap_stages = &[*fullscreen_tri_stage.as_stage_create_info(), *fragment_tonemap_stage.as_stage_create_info()];

        let tonemap_pipeline_desc = tonemap_pipeline_baked.as_pipeline_create_info(
            tonemap_stages,
            tonemap_pipeline_layout,
            render_passes.tonemap,
            0,
        );

        let depth_pre_pass_transmissive_desc = depth_pre_pass_baked.as_pipeline_create_info(
            depth_pre_pass_stage,
            depth_pre_pass_pipeline_layout,
            render_passes.transmission,
            0,
        );

        let depth_pre_pass_transmissive_alpha_clip_desc = depth_pre_pass_alpha_clip_baked
            .as_pipeline_create_info(
                depth_pre_pass_alpha_clip_stages,
                depth_pre_pass_pipeline_layout,
                render_passes.transmission,
                0,
            );

        let cluster_debugging_stages = &[*cluster_debugging_vs_stage.as_stage_create_info(), *cluster_debugging_fs_stage.as_stage_create_info()];

        let cluster_debugging_pipeline_desc = cluster_debugging_baked.as_pipeline_create_info(
            cluster_debugging_stages,
            cluster_debugging_pipeline_layout,
            render_passes.draw,
            0,
        );

        let defer_opaque_stages = &[*defer_vs_stage.as_stage_create_info(), *defer_opaque_stage.as_stage_create_info()];

        let defer_opaque_pipeline_desc = defer_baked.as_pipeline_create_info(
            defer_opaque_stages,
            depth_pre_pass_pipeline_layout,
            render_passes.defer,
            0,
        );

        let defer_alpha_clip_stages = &[*defer_vs_stage.as_stage_create_info(), *defer_alpha_clip_stage.as_stage_create_info()];

        let defer_alpha_clip_pipeline_desc = defer_baked.as_pipeline_create_info(
            defer_alpha_clip_stages,
            depth_pre_pass_pipeline_layout,
            render_passes.defer,
            0,
        );

        let pipeline_descs = [
            *normal_pipeline_desc,
            *depth_pre_pass_desc,
            *depth_pre_pass_alpha_clip_desc,
            *depth_pre_pass_transmissive_desc,
            *depth_pre_pass_transmissive_alpha_clip_desc,
            *transmission_pipeline_desc,
            *tonemap_pipeline_desc,
            *cluster_debugging_pipeline_desc,
            *defer_opaque_pipeline_desc,
            *defer_alpha_clip_pipeline_desc,
        ];

        let pipelines =
            unsafe { device.create_graphics_pipelines(pipeline_cache, &pipeline_descs, None) }
                .map_err(|(_, err)| err)?;

        let mut compute_pipeline_stages = vec![
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*frustum_culling_stage.as_stage_create_info())
                .layout(frustum_culling_pipeline_layout),
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*demultiplex_draws_stage.as_stage_create_info())
                .layout(frustum_culling_pipeline_layout),
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*assign_lights_to_clusters_stage.as_stage_create_info())
                .layout(lights_pipeline_layout),
            *vk::ComputePipelineCreateInfo::builder()
                .stage(*write_cluster_data_stage.as_stage_create_info())
                .layout(write_cluster_data_pipeline_layout),
        ];

        if let Some(stages) = ray_tracing_stages.as_ref() {
            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.acceleration_structure_debugging.as_stage_create_info())
                    .layout(acceleration_structure_debugging_layout),
            );

            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.ray_trace_sun_shadow.as_stage_create_info())
                    .layout(acceleration_structure_debugging_layout),
            );
        }

        let compute_pipelines = unsafe {
            device.create_compute_pipelines(pipeline_cache, &compute_pipeline_stages, None)
        }
        .map_err(|(_, err)| err)?;

        Ok((Self {
            normal: pipelines[0],
            depth_pre_pass: pipelines[1],
            depth_pre_pass_alpha_clip: pipelines[2],
            depth_pre_pass_transmissive: pipelines[3],
            depth_pre_pass_transmissive_alpha_clip: pipelines[4],
            transmission: pipelines[5],
            tonemap: pipelines[6],
            cluster_debugging: pipelines[7],
            defer_opaque: pipelines[8],
            defer_alpha_clip: pipelines[9],
            sun_shadow: None,
            frustum_culling: compute_pipelines[0],
            demultiplex_draws: compute_pipelines[1],
            assign_lights_to_clusters: compute_pipelines[2],
            write_cluster_data: compute_pipelines[3],
            acceleration_structure_debugging: if enable_ray_tracing {
                Some(compute_pipelines[4])
            } else {
                None
            },
            depth_pre_pass_pipeline_layout,
            draw_pipeline_layout,
            tonemap_pipeline_layout,
            transmission_pipeline_layout,
            frustum_culling_pipeline_layout,
            acceleration_structure_debugging_layout,
            lights_pipeline_layout,
            write_cluster_data_pipeline_layout,
            cluster_debugging_pipeline_layout,
        }, descriptor_set_layouts, pool_sizes))
    }
}

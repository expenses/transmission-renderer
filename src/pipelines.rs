use crate::descriptor_sets::DescriptorSetLayouts;
use crate::render_passes::RenderPasses;
use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::vk;
use glam::{Vec2, Vec3};
use std::path::{Path, PathBuf};

fn rgba_flags() -> vk::ColorComponentFlags {
    vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A
}

fn read_shader(parent: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let mut path = parent.join(name);
    path.set_extension("spv");
    Ok(std::fs::read(&path)?)
}

#[derive(Clone, Copy)]
pub(crate) struct RayTracingPipelines {
    pub acceleration_structure_debugging: vk::Pipeline,
    pub ray_trace_sun_shadow: vk::Pipeline,
    pub reconstruct_shadow_buffer: vk::Pipeline,
    pub tile_classification: vk::Pipeline,
    pub filter_pass_0: vk::Pipeline,
    pub filter_pass_1: vk::Pipeline,
    pub filter_pass_2: vk::Pipeline,
}

pub(crate) struct Pipelines {
    pub normal: vk::Pipeline,
    pub depth_pre_pass: vk::Pipeline,
    pub depth_pre_pass_alpha_clip: vk::Pipeline,
    pub depth_pre_pass_transmissive: vk::Pipeline,
    pub depth_pre_pass_transmissive_alpha_clip: vk::Pipeline,
    pub transmission: vk::Pipeline,
    pub tonemap: vk::Pipeline,
    pub frustum_culling: vk::Pipeline,
    pub demultiplex_draws: vk::Pipeline,
    pub assign_lights_to_clusters: vk::Pipeline,
    pub write_cluster_data: vk::Pipeline,
    pub cluster_debugging: vk::Pipeline,
    pub defer_opaque: vk::Pipeline,
    pub defer_alpha_clip: vk::Pipeline,
    pub ray_tracing_pipelines: Option<RayTracingPipelines>,
    pub depth_pre_pass_pipeline_layout: vk::PipelineLayout,
    pub draw_pipeline_layout: vk::PipelineLayout,
    pub tonemap_pipeline_layout: vk::PipelineLayout,
    pub transmission_pipeline_layout: vk::PipelineLayout,
    pub frustum_culling_pipeline_layout: vk::PipelineLayout,
    pub acceleration_structure_debugging_layout: vk::PipelineLayout,
    pub lights_pipeline_layout: vk::PipelineLayout,
    pub write_cluster_data_pipeline_layout: vk::PipelineLayout,
    pub cluster_debugging_pipeline_layout: vk::PipelineLayout,
    pub ray_trace_sun_shadow_layout: vk::PipelineLayout,
    pub reconstruct_shadow_buffer_layout: vk::PipelineLayout,
    pub tile_classification_layout: vk::PipelineLayout,
    pub filter_pass_layout: vk::PipelineLayout,
}

impl Pipelines {
    pub fn new(
        device: &ash::Device,
        debug_utils_loader: &DebugUtilsLoader,
        render_passes: &RenderPasses,
        pipeline_cache: vk::PipelineCache,
        enable_ray_tracing: bool,
    ) -> anyhow::Result<(Self, DescriptorSetLayouts)> {
        let _span = tracy_client::span!("Pipelines::new");

        let normal = &PathBuf::from("compiled-shaders/normal");
        let ray_tracing = &PathBuf::from("compiled-shaders/ray-tracing");

        let maybe_ray_tracing = if enable_ray_tracing {
            ray_tracing
        } else {
            normal
        };

        let mut layouts = ash_reflect::DescriptorSetLayouts::default();

        struct RayTracingStages {
            acceleration_structure_debugging: ash_reflect::ShaderModule,
            ray_trace_sun_shadow: ash_reflect::ShaderModule,
            reconstruct_shadow_buffer: ash_reflect::ShaderModule,
            tile_classification: ash_reflect::ShaderModule,
            filter_pass_0: ash_reflect::ShaderModule,
            filter_pass_1: ash_reflect::ShaderModule,
            filter_pass_2: ash_reflect::ShaderModule,
        }

        let ray_tracing_stages = if enable_ray_tracing {
            Some(RayTracingStages {
                acceleration_structure_debugging: layouts.load_and_merge_module(
                    device,
                    &read_shader(ray_tracing, "debugging_acceleration_structure_debugging")?,
                )?,
                ray_trace_sun_shadow: layouts.load_and_merge_module(
                    device,
                    &read_shader(ray_tracing, "ray_trace_sun_shadow")?,
                )?,
                reconstruct_shadow_buffer: layouts.load_and_merge_module(
                    device,
                    &read_shader(ray_tracing, "reconstruct_shadow_buffer")?,
                )?,
                tile_classification: layouts.load_and_merge_module(
                    device,
                    &read_shader(ray_tracing, "tile_classification")?,
                )?,
                filter_pass_0: layouts
                    .load_and_merge_module(device, &read_shader(ray_tracing, "filter_pass_0")?)?,
                filter_pass_1: layouts
                    .load_and_merge_module(device, &read_shader(ray_tracing, "filter_pass_1")?)?,
                filter_pass_2: layouts
                    .load_and_merge_module(device, &read_shader(ray_tracing, "filter_pass_2")?)?,
            })
        } else {
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(&read_shader(
                ray_tracing,
                "debugging_acceleration_structure_debugging",
            )?)?);
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(&read_shader(
                ray_tracing,
                "ray_trace_sun_shadow",
            )?)?);
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(&read_shader(
                ray_tracing,
                "reconstruct_shadow_buffer",
            )?)?);
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(&read_shader(
                ray_tracing,
                "tile_classification",
            )?)?);
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(&read_shader(
                ray_tracing,
                "filter_pass_0",
            )?)?);
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(&read_shader(
                ray_tracing,
                "filter_pass_1",
            )?)?);
            layouts.merge_from_reflection(&ash_reflect::ShaderReflection::new(&read_shader(
                ray_tracing,
                "filter_pass_2",
            )?)?);

            None
        };

        let fragment_stage = layouts
            .load_and_merge_module(device, &read_shader(maybe_ray_tracing, "fragment_opaque")?)?;

        let fragment_transmission_stage = layouts.load_and_merge_module(
            device,
            &read_shader(maybe_ray_tracing, "fragment_transmission")?,
        )?;

        let vertex_instanced_stage =
            layouts.load_and_merge_module(device, &read_shader(normal, "vertex_instanced")?)?;

        let vertex_instanced_with_scale_stage = layouts
            .load_and_merge_module(device, &read_shader(normal, "vertex_instanced_with_scale")?)?;

        let vertex_depth_pre_pass_stage = layouts
            .load_and_merge_module(device, &read_shader(normal, "depth_pre_pass_vertex")?)?;

        let vertex_depth_pre_pass_alpha_clip_stage = layouts.load_and_merge_module(
            device,
            &read_shader(normal, "depth_pre_pass_vertex_alpha_clip")?,
        )?;

        let fragment_depth_pre_pass_alpha_clip_stage = layouts.load_and_merge_module(
            device,
            &read_shader(normal, "depth_pre_pass_fragment_alpha_clip")?,
        )?;

        let fullscreen_tri_stage = layouts
            .load_and_merge_module(device, &read_shader(normal, "vertex_fullscreen_tri")?)?;

        let fragment_tonemap_stage =
            layouts.load_and_merge_module(device, &read_shader(normal, "fragment_tonemap")?)?;

        let frustum_culling_stage =
            layouts.load_and_merge_module(device, &read_shader(normal, "frustum_culling")?)?;

        let demultiplex_draws_stage =
            layouts.load_and_merge_module(device, &read_shader(normal, "demultiplex_draws")?)?;

        let assign_lights_to_clusters_stage = layouts
            .load_and_merge_module(device, &read_shader(normal, "assign_lights_to_clusters")?)?;

        let write_cluster_data_stage =
            layouts.load_and_merge_module(device, &read_shader(normal, "write_cluster_data")?)?;

        let cluster_debugging_vs_stage = layouts.load_and_merge_module(
            device,
            &read_shader(normal, "debugging_cluster_debugging_vs")?,
        )?;

        let cluster_debugging_fs_stage = layouts.load_and_merge_module(
            device,
            &read_shader(normal, "debugging_cluster_debugging_fs")?,
        )?;

        let defer_opaque_stage = layouts
            .load_and_merge_module(device, &read_shader(normal, "deferred_defer_opaque")?)?;

        let defer_alpha_clip_stage = layouts
            .load_and_merge_module(device, &read_shader(normal, "deferred_defer_alpha_clip")?)?;

        let defer_vs_stage =
            layouts.load_and_merge_module(device, &read_shader(normal, "deferred_vs")?)?;

        let settings = ash_reflect::Settings {
            max_unbounded_descriptors: crate::MAX_IMAGES,
            enable_partially_bound_unbounded_descriptors: true,
        };

        let layouts = layouts.build(device, settings)?;
        let descriptor_set_layouts =
            DescriptorSetLayouts::from_reflected_layouts(device, debug_utils_loader, layouts)?;
        let layouts = &descriptor_set_layouts.layouts;

        let draw_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &layouts
                    .pipeline_layout_for_shaders(&["fragment::opaque", "vertex::instanced"])?
                    .as_ref(),
                None,
            )?
        };

        let tile_classification_layout = unsafe {
            device.create_pipeline_layout(
                &layouts
                    .pipeline_layout_for_shaders(&["tile_classification"])?
                    .as_ref(),
                None,
            )?
        };

        let filter_pass_layout = unsafe {
            device.create_pipeline_layout(
                &layouts
                    .pipeline_layout_for_shaders(&["filter_pass_0", "filter_pass_2"])?
                    .as_ref(),
                None,
            )?
        };

        let reconstruct_shadow_buffer_layout = unsafe {
            device.create_pipeline_layout(
                &layouts
                    .pipeline_layout_for_shaders(&["reconstruct_shadow_buffer"])?
                    .as_ref(),
                None,
            )?
        };

        let lights_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &layouts
                    .pipeline_layout_for_shaders(&["assign_lights_to_clusters"])?
                    .as_ref(),
                None,
            )
        }?;

        let ray_trace_sun_shadow_layout = unsafe {
            device.create_pipeline_layout(
                &layouts
                    .pipeline_layout_for_shaders(&["ray_trace_sun_shadow"])?
                    .as_ref(),
                None,
            )
        }?;

        let depth_pre_pass_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        *descriptor_set_layouts.main,
                        *descriptor_set_layouts.instance_buffer,
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
                        *descriptor_set_layouts.main,
                        *descriptor_set_layouts.instance_buffer,
                        *descriptor_set_layouts.lights,
                        *descriptor_set_layouts.single_sampled_image,
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
                        *descriptor_set_layouts.frustum_culling,
                        *descriptor_set_layouts.instance_buffer,
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
                        *descriptor_set_layouts.main,
                        *descriptor_set_layouts.single_sampled_image,
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
                        *descriptor_set_layouts.main,
                        *descriptor_set_layouts.acceleration_structure_debugging,
                        *descriptor_set_layouts.instance_buffer,
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
                        *descriptor_set_layouts.main,
                        *descriptor_set_layouts.cluster_data,
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

        let cluster_debugging_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[*descriptor_set_layouts.cluster_data])
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
                    .color_write_mask(rgba_flags())
                    .blend_enable(false),
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(rgba_flags())
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
            colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(rgba_flags())
                .blend_enable(false)],
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
                .color_write_mask(rgba_flags())
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
                .color_write_mask(rgba_flags())
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
                    .color_write_mask(rgba_flags())
                    .blend_enable(false),
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(rgba_flags())
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

        let stages = &[
            *vertex_instanced_stage.as_stage_create_info(),
            *fragment_stage.as_stage_create_info(),
        ];

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

        let tonemap_stages = &[
            *fullscreen_tri_stage.as_stage_create_info(),
            *fragment_tonemap_stage.as_stage_create_info(),
        ];

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

        let cluster_debugging_stages = &[
            *cluster_debugging_vs_stage.as_stage_create_info(),
            *cluster_debugging_fs_stage.as_stage_create_info(),
        ];

        let cluster_debugging_pipeline_desc = cluster_debugging_baked.as_pipeline_create_info(
            cluster_debugging_stages,
            cluster_debugging_pipeline_layout,
            render_passes.draw,
            0,
        );

        let defer_opaque_stages = &[
            *defer_vs_stage.as_stage_create_info(),
            *defer_opaque_stage.as_stage_create_info(),
        ];

        let defer_opaque_pipeline_desc = defer_baked.as_pipeline_create_info(
            defer_opaque_stages,
            depth_pre_pass_pipeline_layout,
            render_passes.defer,
            0,
        );

        let defer_alpha_clip_stages = &[
            *defer_vs_stage.as_stage_create_info(),
            *defer_alpha_clip_stage.as_stage_create_info(),
        ];

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
                    .stage(
                        *stages
                            .acceleration_structure_debugging
                            .as_stage_create_info(),
                    )
                    .layout(acceleration_structure_debugging_layout),
            );

            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.ray_trace_sun_shadow.as_stage_create_info())
                    .layout(ray_trace_sun_shadow_layout),
            );

            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.reconstruct_shadow_buffer.as_stage_create_info())
                    .layout(reconstruct_shadow_buffer_layout),
            );

            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.tile_classification.as_stage_create_info())
                    .layout(tile_classification_layout),
            );

            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.filter_pass_0.as_stage_create_info())
                    .layout(filter_pass_layout),
            );

            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.filter_pass_1.as_stage_create_info())
                    .layout(filter_pass_layout),
            );

            compute_pipeline_stages.push(
                *vk::ComputePipelineCreateInfo::builder()
                    .stage(*stages.filter_pass_2.as_stage_create_info())
                    .layout(filter_pass_layout),
            );
        }

        let compute_pipelines = unsafe {
            device.create_compute_pipelines(pipeline_cache, &compute_pipeline_stages, None)
        }
        .map_err(|(_, err)| err)?;

        Ok((
            Self {
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
                frustum_culling: compute_pipelines[0],
                demultiplex_draws: compute_pipelines[1],
                assign_lights_to_clusters: compute_pipelines[2],
                write_cluster_data: compute_pipelines[3],
                ray_tracing_pipelines: if enable_ray_tracing {
                    Some(RayTracingPipelines {
                        acceleration_structure_debugging: compute_pipelines[4],
                        ray_trace_sun_shadow: compute_pipelines[5],
                        reconstruct_shadow_buffer: compute_pipelines[6],
                        tile_classification: compute_pipelines[7],
                        filter_pass_0: compute_pipelines[8],
                        filter_pass_1: compute_pipelines[9],
                        filter_pass_2: compute_pipelines[10],
                    })
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
                ray_trace_sun_shadow_layout,
                reconstruct_shadow_buffer_layout,
                tile_classification_layout,
                filter_pass_layout,
            },
            descriptor_set_layouts,
        ))
    }
}

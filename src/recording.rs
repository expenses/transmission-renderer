use crate::*;

pub(crate) struct RecordParams<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub pipelines: &'a Pipelines,
    pub draw_buffers: &'a DrawBuffers,
    pub light_buffers: &'a LightBuffers,
    pub profiling_ctx: &'a ProfilingContext,
    pub model_buffers: &'a ModelBuffers,
    pub render_passes: &'a RenderPasses,
    pub draw_framebuffer: vk::Framebuffer,
    pub depth_framebuffer: vk::Framebuffer,
    pub sun_shadow_framebuffer: vk::Framebuffer,
    pub transmission_framebuffer: vk::Framebuffer,
    pub tonemap_framebuffer: vk::Framebuffer,
    pub hdr_framebuffer: &'a ash_abstractions::Image,
    pub opaque_sampled_hdr_framebuffer: &'a ash_abstractions::Image,
    pub descriptor_sets: &'a DescriptorSets,
    pub dynamic: DynamicRecordParams,
    pub toggle: bool,
}

pub(crate) struct DynamicRecordParams {
    pub extent: vk::Extent2D,
    pub num_primitives: u32,
    pub num_instances: u32,
    pub num_lights: u32,
    pub push_constants: PushConstants,
    pub view_matrix: Mat4,
    pub perspective_matrix: Mat4,
    pub opaque_mip_levels: u32,
    pub tonemapping_params: colstodian::tonemap::BakedLottesTonemapperParams,
    pub camera_rotation: Quat,
}

pub(crate) unsafe fn record(params: RecordParams) -> anyhow::Result<()> {
    let _span = tracy_client::span!("Command buffer recording");

    let RecordParams {
        device,
        command_buffer,
        pipelines,
        draw_buffers,
        light_buffers,
        profiling_ctx,
        render_passes,
        draw_framebuffer,
        depth_framebuffer,
        sun_shadow_framebuffer,
        transmission_framebuffer,
        tonemap_framebuffer,
        toggle,
        dynamic:
            DynamicRecordParams {
                num_primitives,
                num_instances,
                num_lights,
                push_constants,
                view_matrix,
                perspective_matrix,
                extent,
                opaque_mip_levels,
                tonemapping_params,
                camera_rotation,
            },
        model_buffers,
        hdr_framebuffer,
        opaque_sampled_hdr_framebuffer,
        descriptor_sets,
    } = params;

    let depth_pre_pass_clear_values = [
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        },
    ];

    let draw_clear_values = [
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        },
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
    ];

    let black_clear_value = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];

    let area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };

    let draw_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.draw)
        .framebuffer(draw_framebuffer)
        .render_area(area)
        .clear_values(&draw_clear_values);

    let depth_pre_pass_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.depth_pre_pass)
        .framebuffer(depth_framebuffer)
        .render_area(area)
        .clear_values(&depth_pre_pass_clear_values);

    let sun_shadow_clear_values = [
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        },
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0; 4],
            },
        }
    ];

    let sun_shadow_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.sun_shadow)
        .framebuffer(sun_shadow_framebuffer)
        .render_area(area)
        .clear_values(&sun_shadow_clear_values);

    let transmission_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.transmission)
        .framebuffer(transmission_framebuffer)
        .render_area(area);

    let tonemap_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.tonemap)
        .framebuffer(tonemap_framebuffer)
        .render_area(area)
        .clear_values(&black_clear_value);

    let viewport = *vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(extent.width as f32)
        .height(extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    profiling_ctx.reset(device, command_buffer);

    let all_commands_profiling_zone = profiling_zone!(
        "all commands",
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        device,
        command_buffer,
        profiling_ctx
    );

    {
        let _profiling_zone = profiling_zone!(
            "frustum culling",
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            device,
            command_buffer,
            profiling_ctx
        );

        {
            let profiling_zone = profiling_zone!(
                "zeroing the instance count buffer",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_fill_buffer(
                command_buffer,
                draw_buffers.instance_count_buffer.buffer,
                0,
                vk::WHOLE_SIZE,
                0,
            );

            drop(profiling_zone);

            let profiling_zone = profiling_zone!(
                "zeroing the draw count buffer",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_fill_buffer(
                command_buffer,
                draw_buffers.draw_counts_buffer.buffer,
                0,
                vk::WHOLE_SIZE,
                0,
            );

            device.cmd_fill_buffer(
                command_buffer,
                light_buffers.cluster_light_counts.buffer,
                0,
                vk::WHOLE_SIZE,
                0,
            );

            drop(profiling_zone);
        }

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferWrite],
                next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
            }),
            &[],
            &[],
        );

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.frustum_culling_pipeline_layout,
            0,
            &[
                descriptor_sets.frustum_culling,
                descriptor_sets.instance_buffer,
            ],
            &[],
        );

        let trunc_row = |index| perspective_matrix.row(index).truncate();

        // Get the left and top planes (the ones that satisfy 'x + w < 0' and 'y + w < 0') (I think, don't quote me on this)
        // https://github.com/zeux/niagara/blob/98f5d5ae2b48e15e145e3ad13ae7f4f9f1e0e297/src/niagara.cpp#L822-L823
        let frustum_x = (trunc_row(3) + trunc_row(0)).normalize();
        let frustum_y = (trunc_row(3) + trunc_row(1)).normalize();

        device.cmd_push_constants(
            command_buffer,
            pipelines.frustum_culling_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytes_of(&shared_structs::CullingPushConstants {
                view: view_matrix,
                frustum_x_xz: frustum_x.xz(),
                frustum_y_yz: frustum_y.yz(),
                z_near: Z_NEAR,
            }),
        );

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.frustum_culling,
        );

        {
            let _profiling_zone = profiling_zone!(
                "frustum culling compute shader",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_dispatch(command_buffer, dispatch_count(num_instances, 64), 1, 1);
        }

        {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.assign_lights_to_clusters,
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.lights_pipeline_layout,
                0,
                &[descriptor_sets.lights, descriptor_sets.cluster_data],
                &[],
            );

            device.cmd_push_constants(
                command_buffer,
                pipelines.lights_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&shared_structs::AssignLightsPushConstants {
                    view_matrix,
                    view_rotation: camera_rotation.inverse(),
                }),
            );

            device.cmd_dispatch(
                command_buffer,
                dispatch_count(NUM_CLUSTERS, 8),
                dispatch_count(num_lights, 8),
                1,
            );
        }

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther, vk_sync::AccessType::FragmentShaderReadOther],
            }),
            &[],
            &[],
        );

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.demultiplex_draws,
        );

        {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.frustum_culling_pipeline_layout,
                0,
                &[
                    descriptor_sets.frustum_culling,
                    descriptor_sets.instance_buffer,
                ],
                &[],
            );

            let _profiling_zone = profiling_zone!(
                "demultiplex draws compute shader",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_dispatch(command_buffer, dispatch_count(num_primitives, 64), 1, 1);
        }

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[vk_sync::AccessType::IndirectBuffer],
            }),
            &[],
            &[],
        );
    }

    device.cmd_begin_render_pass(
        command_buffer,
        &depth_pre_pass_render_pass_info,
        vk::SubpassContents::INLINE,
    );

    device.cmd_set_scissor(command_buffer, 0, &[area]);
    device.cmd_set_viewport(command_buffer, 0, &[viewport]);

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.depth_pre_pass_pipeline_layout,
        0,
        &[
            descriptor_sets.main,
            descriptor_sets.instance_buffer,
            descriptor_sets.lights,
        ],
        &[],
    );

    device.cmd_push_constants(
        command_buffer,
        pipelines.depth_pre_pass_pipeline_layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        bytes_of(&push_constants),
    );

    device.cmd_bind_vertex_buffers(
        command_buffer,
        0,
        &[
            model_buffers.position.buffer,
            model_buffers.normal.buffer,
            model_buffers.uv.buffer,
        ],
        &[0, 0, 0],
    );

    device.cmd_bind_index_buffer(
        command_buffer,
        model_buffers.index.buffer,
        0,
        vk::IndexType::UINT32,
    );

    {
        let _depth_profiling_zone =
            profiling_zone!("depth pre pass", device, command_buffer, profiling_ctx);

        {
            let _profiling_zone = profiling_zone!(
                "depth pre pass opaque",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass,
            );

            draw_buffers
                .opaque
                .record(device, &draw_buffers.draw_counts_buffer, 0, command_buffer);
        }

        {
            let _profiling_zone = profiling_zone!(
                "depth pre pass alpha clipped",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass_alpha_clip,
            );

            draw_buffers.alpha_clip.record(
                device,
                &draw_buffers.draw_counts_buffer,
                1,
                command_buffer,
            );
        }
    }

    device.cmd_end_render_pass(command_buffer);

    device.cmd_begin_render_pass(
        command_buffer,
        &sun_shadow_render_pass_info,
        vk::SubpassContents::INLINE
    );

    if let Some(pipeline) = pipelines.sun_shadow {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline,
        );

        {
            let _profiling_zone = profiling_zone!("sun shadow opaque", device, command_buffer, profiling_ctx);
            draw_buffers
                .opaque
                .record(device, &draw_buffers.draw_counts_buffer, 0, command_buffer);
        }

        {
            let _profiling_zone =
                profiling_zone!("sun shadow alpha clipped", device, command_buffer, profiling_ctx);
            draw_buffers
                .alpha_clip
                .record(device, &draw_buffers.draw_counts_buffer, 1, command_buffer);
        }
    }

    device.cmd_end_render_pass(command_buffer);

    device.cmd_begin_render_pass(
        command_buffer,
        &draw_render_pass_info,
        vk::SubpassContents::INLINE,
    );

    /*{
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipelines.cluster_debugging,
        );

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipelines.cluster_debugging_pipeline_layout,
            0,
            &[
                descriptor_sets.cluster_data,
            ],
            &[],
        );

        device.cmd_draw(command_buffer, NUM_CLUSTERS * 8, 1, 0, 0);
    }*/

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.normal,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.pipeline_layout,
        0,
        &[
            descriptor_sets.main,
            descriptor_sets.instance_buffer,
            descriptor_sets.lights,
            descriptor_sets.sun_shadow_buffer,
        ],
        &[],
    );

    {
        let _profiling_zone = profiling_zone!("main opaque", device, command_buffer, profiling_ctx);
        draw_buffers
            .opaque
            .record(device, &draw_buffers.draw_counts_buffer, 0, command_buffer);
    }

    {
        let _profiling_zone =
            profiling_zone!("main alpha clipped", device, command_buffer, profiling_ctx);
        draw_buffers
            .alpha_clip
            .record(device, &draw_buffers.draw_counts_buffer, 1, command_buffer);
    }

    device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);

    {
        let _profiling_zone = profiling_zone!(
            "depth pre pass transmissive",
            device,
            command_buffer,
            profiling_ctx
        );

        {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass_transmissive,
            );

            draw_buffers.transmission.record(
                device,
                &draw_buffers.draw_counts_buffer,
                2,
                command_buffer,
            );
        }

        {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass_transmissive_alpha_clip,
            );

            draw_buffers.transmission_alpha_clip.record(
                device,
                &draw_buffers.draw_counts_buffer,
                3,
                command_buffer,
            );
        }
    }

    device.cmd_end_render_pass(command_buffer);

    {
        let _profiling_zone = profiling_zone!(
            "opaque framebuffer mipchain",
            device,
            command_buffer,
            profiling_ctx
        );

        ash_abstractions::generate_mips(
            device,
            command_buffer,
            opaque_sampled_hdr_framebuffer.image,
            extent.width as i32,
            extent.height as i32,
            opaque_mip_levels,
            &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
            vk_sync::ImageLayout::Optimal,
        );
    }

    device.cmd_begin_render_pass(
        command_buffer,
        &transmission_render_pass_info,
        vk::SubpassContents::INLINE,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.transmission_pipeline_layout,
        0,
        &[
            descriptor_sets.main,
            descriptor_sets.instance_buffer,
            descriptor_sets.lights,
            descriptor_sets.opaque_sampled_hdr_framebuffer,
        ],
        &[],
    );

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.transmission,
    );

    {
        let _profiling_zone = profiling_zone!(
            "opaque transmissive objects",
            device,
            command_buffer,
            profiling_ctx
        );

        draw_buffers.transmission.record(
            device,
            &draw_buffers.draw_counts_buffer,
            2,
            command_buffer,
        );
    }

    {
        let _profiling_zone = profiling_zone!(
            "alpha clip transmissive objects",
            device,
            command_buffer,
            profiling_ctx
        );

        draw_buffers.transmission_alpha_clip.record(
            device,
            &draw_buffers.draw_counts_buffer,
            3,
            command_buffer,
        );
    }

    device.cmd_end_render_pass(command_buffer);

    if let Some(pipeline) = pipelines
        .acceleration_structure_debugging
        .filter(|_| toggle)
    {
        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: hdr_framebuffer.image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.acceleration_structure_debugging_layout,
            0,
            &[descriptor_sets.main, descriptor_sets.acceleration_structure_debugging, descriptor_sets.instance_buffer],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            pipelines.acceleration_structure_debugging_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytes_of(&push_constants),
        );

        device.cmd_dispatch(
            command_buffer,
            dispatch_count(extent.width, 8),
            dispatch_count(extent.height, 8),
            1,
        );

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: hdr_framebuffer.image,
                range: subresource_range,
                ..Default::default()
            }],
        );
    }

    device.cmd_begin_render_pass(
        command_buffer,
        &tonemap_render_pass_info,
        vk::SubpassContents::INLINE,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.tonemap_pipeline_layout,
        0,
        &[descriptor_sets.main, descriptor_sets.hdr_framebuffer],
        &[],
    );

    device.cmd_push_constants(
        command_buffer,
        pipelines.tonemap_pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        bytes_of(&tonemapping_params),
    );

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.tonemap,
    );

    {
        let _profiling_zone = profiling_zone!("tonemapping", device, command_buffer, profiling_ctx);

        device.cmd_draw(command_buffer, 3, 1, 0, 0);
    }

    device.cmd_end_render_pass(command_buffer);

    {
        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(1)
            .level_count(opaque_mip_levels - 1)
            .layer_count(1);

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_accesses: &[vk_sync::AccessType::TransferWrite],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: opaque_sampled_hdr_framebuffer.image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );
    }

    drop(all_commands_profiling_zone);

    Ok(())
}

pub(crate) unsafe fn record_write_cluster_data(
    init_resources: &ash_abstractions::InitResources,
    pipelines: &Pipelines,
    descriptor_sets: &DescriptorSets,
    perspective_matrix: Mat4,
    screen_dimensions: UVec2,
) {
    init_resources.device.cmd_bind_pipeline(
        init_resources.command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipelines.write_cluster_data,
    );

    init_resources.device.cmd_bind_descriptor_sets(
        init_resources.command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipelines.write_cluster_data_pipeline_layout,
        0,
        &[descriptor_sets.main, descriptor_sets.cluster_data],
        &[],
    );

    init_resources.device.cmd_push_constants(
        init_resources.command_buffer,
        pipelines.write_cluster_data_pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes_of(&shared_structs::WriteClusterDataPushConstants {
            inverse_perspective: perspective_matrix.inverse(),
            screen_dimensions,
        }),
    );

    init_resources.device.cmd_dispatch(
        init_resources.command_buffer,
        dispatch_count(NUM_CLUSTERS_X, 4),
        dispatch_count(NUM_CLUSTERS_Y, 4),
        dispatch_count(NUM_DEPTH_SLICES, 4),
    );
}

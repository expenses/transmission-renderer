use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{Surface as SurfaceLoader, Swapchain as SwapchainLoader};
use ash::vk;
use ash_abstractions::CStrList;
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

use glam::{Mat4, UVec2, Vec2, Vec3, Vec3A, Vec4};
use shared_structs::PointLight;

fn perspective_infinite_z_vk(vertical_fov: f32, aspect_ratio: f32, z_near: f32) -> Mat4 {
    let t = (vertical_fov / 2.0).tan();
    let sy = 1.0 / t;
    let sx = sy / aspect_ratio;

    Mat4::from_cols(
        Vec4::new(sx, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -sy, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0, -1.0),
        Vec4::new(0.0, 0.0, -z_near, 0.0),
    )
}

fn main() -> anyhow::Result<()> {
    {
        use simplelog::*;

        CombinedLogger::init(vec![TermLogger::new(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        )])?;
    }

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    let api_version = vk::API_VERSION_1_1;

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Nice Grass\0")?)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    let instance_extensions = CStrList::new({
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window)?;
        instance_extensions.extend(&[
            DebugUtilsLoader::name(),
            vk::KhrGetPhysicalDeviceProperties2Fn::name(),
        ]);
        instance_extensions
    });

    let enabled_layers = CStrList::new(vec![CStr::from_bytes_with_nul(
        b"VK_LAYER_KHRONOS_validation\0",
    )?]);

    let device_extensions = CStrList::new(vec![SwapchainLoader::name()]);

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(ash_abstractions::vulkan_debug_utils_callback));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(instance_extensions.pointers())
                .enabled_layer_names(enabled_layers.pointers())
                .push_next(&mut debug_messenger_info),
            None,
        )
    }?;

    let debug_utils_loader = DebugUtilsLoader::new(&entry, &instance);

    let surface_loader = SurfaceLoader::new(&entry, &instance);

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;

    let (physical_device, graphics_queue_family, surface_format) =
        match ash_abstractions::select_physical_device(
            &instance,
            &device_extensions,
            &surface_loader,
            surface,
            vk::Format::B8G8R8A8_SRGB,
        )? {
            Some(selection) => selection,
            None => {
                log::info!("No suitable device found 💔. Exiting program");
                return Ok(());
            }
        };

    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;

    let device = {
        let queue_info = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_queue_family)
            .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::builder();

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(device_extensions.pointers())
            .enabled_layer_names(enabled_layers.pointers());

        unsafe { instance.create_device(physical_device, &device_info, None) }?
    };

    let render_passes = RenderPasses::new(&device, surface_format.format)?;

    let lights_dsl = unsafe {
        device.create_descriptor_set_layout(
            &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ]),
            None,
        )
    }?;

    let pipeline_cache =
        unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }?;

    let pipelines = Pipelines::new(&device, &render_passes, lights_dsl, pipeline_cache)?;

    let queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

    let (command_buffer, command_pool) = {
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder().queue_family_index(graphics_queue_family),
                None,
            )
        }?;

        let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }?[0];

        (command_buffer, command_pool)
    };

    let mut allocator =
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
        })?;

    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
    }?;

    let mut init_resources = ash_abstractions::InitResources {
        command_buffer,
        device: &device,
        allocator: &mut allocator,
        debug_utils_loader: Some(&debug_utils_loader),
    };

    let window_size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: window_size.width,

        height: window_size.height,
    };

    let filename = std::env::args().nth(1).unwrap();

    let (model_vertices, model_indices, model_num_indices) =
        load_gltf_from_bytes(&std::fs::read(&filename)?, &mut init_resources)?;

    let mut depthbuffer = create_depthbuffer(extent.width, extent.height, &mut init_resources)?;

    let lights_buffer = ash_abstractions::Buffer::new(
        unsafe {
            cast_slice(&[
                PointLight {
                    position: Vec3::new(0.0, 10.0, 0.0).into(),
                    colour_and_intensity: Vec4::new(1.0, 0.0, 0.0, 10000.0),
                },
                PointLight {
                    position: Vec3::new(1000.0, 10.0, 0.0).into(),
                    colour_and_intensity: Vec4::new(0.0, 0.0, 1.0, 10000.0),
                },
            ])
        },
        "lights",
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &mut init_resources,
    )?;

    let tonemapping_params_buffer = ash_abstractions::Buffer::new(
        unsafe {
            bytes_of(&colstodian::tonemap::BakedLottesTonemapperParams::from(
                colstodian::tonemap::LottesTonemapperParams {
                    ..Default::default()
                },
            ))
        },
        "tonemapping params",
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        &mut init_resources,
    )?;

    drop(init_resources);

    unsafe {
        device.end_command_buffer(command_buffer)?;
        let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

        device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
            fence,
        )?;

        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    // Swapchain

    let mut image_count = (surface_caps.min_image_count + 1).max(3);
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    log::info!("Using {} swapchain images at a time.", image_count);

    let swapchain_loader = SwapchainLoader::new(&instance, &device);

    let mut swapchain_info = *vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let mut swapchain =
        ash_abstractions::Swapchain::new(&device, &swapchain_loader, swapchain_info)?;

    let mut swapchain_image_framebuffers = create_swapchain_image_framebuffers(
        &device,
        extent,
        &swapchain,
        &render_passes,
        &depthbuffer,
    )?;

    let mut keyboard_state = KeyboardState::default();

    let mut camera = dolly::rig::CameraRig::builder()
        .with(dolly::drivers::Position::new(Vec3::new(2.0, 1000.0, 1.0)))
        .with(dolly::drivers::YawPitch::new().pitch_degrees(-74.0))
        .with(dolly::drivers::Smooth::new_position_rotation(0.5, 0.25))
        .build();

    let mut perspective_matrix = perspective_infinite_z_vk(
        59.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        0.1,
    );

    let num_tiles = UVec2::splat(16);

    let mut push_constants = shared_structs::PushConstants {
        // Updated every frame.
        proj_view: Default::default(),
        view_position: Default::default(),
        sun_dir: Vec3::new(1.0, 10.0, 1.0).normalize().into(),
        sun_intensity: Vec3::ONE.into(),
        num_tiles,
        tile_size_in_pixels: Vec2::new(extent.width as f32, extent.height as f32)
            / num_tiles.as_vec2(),
    };

    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&[
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1),
                ])
                .max_sets(1),
            None,
        )
    }?;

    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[lights_dsl])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    let lights_ds = descriptor_sets[0];

    unsafe {
        device.update_descriptor_sets(
            &[
                *vk::WriteDescriptorSet::builder()
                    .dst_set(lights_ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[*vk::DescriptorBufferInfo::builder()
                        .buffer(lights_buffer.buffer)
                        .range(vk::WHOLE_SIZE)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(lights_ds)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[*vk::DescriptorBufferInfo::builder()
                        .buffer(tonemapping_params_buffer.buffer)
                        .range(vk::WHOLE_SIZE)]),
            ],
            &[],
        )
    }

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let present_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
    let render_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
    let render_fence = unsafe { device.create_fence(&fence_info, None)? };

    let mut cursor_grab = false;

    let mut screen_center =
        winit::dpi::LogicalPosition::new(extent.width as f64 / 2.0, extent.height as f64 / 2.0);

    event_loop.run(move |event, _, control_flow| {
        let loop_closure = || -> anyhow::Result<()> {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(key),
                                ..
                            },
                        ..
                    } => {
                        let is_pressed = state == ElementState::Pressed;

                        match key {
                            VirtualKeyCode::W => keyboard_state.forwards = is_pressed,
                            VirtualKeyCode::S => keyboard_state.backwards = is_pressed,
                            VirtualKeyCode::A => keyboard_state.left = is_pressed,
                            VirtualKeyCode::D => keyboard_state.right = is_pressed,
                            VirtualKeyCode::F11 => {
                                if is_pressed {
                                    if window.fullscreen().is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                                    }
                                }
                            }
                            VirtualKeyCode::G => {
                                if is_pressed {
                                    cursor_grab = !cursor_grab;

                                    if cursor_grab {
                                        window.set_cursor_position(screen_center)?;
                                    }

                                    window.set_cursor_visible(!cursor_grab);
                                    window.set_cursor_grab(cursor_grab)?;
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if cursor_grab {
                            let position = position.to_logical::<f64>(window.scale_factor());

                            window.set_cursor_position(screen_center)?;

                            camera
                                .driver_mut::<dolly::drivers::YawPitch>()
                                .rotate_yaw_pitch(
                                    0.1 * (screen_center.x - position.x) as f32,
                                    0.1 * (screen_center.y - position.y) as f32,
                                );
                        }
                    }
                    WindowEvent::Resized(new_size) => {
                        extent.width = new_size.width;
                        extent.height = new_size.height;

                        push_constants.tile_size_in_pixels =
                            Vec2::new(extent.width as f32, extent.height as f32)
                                / num_tiles.as_vec2();

                        perspective_matrix = perspective_infinite_z_vk(
                            59.0_f32.to_radians(),
                            extent.width as f32 / extent.height as f32,
                            0.1,
                        );

                        screen_center = winit::dpi::LogicalPosition::new(
                            extent.width as f64 / 2.0,
                            extent.height as f64 / 2.0,
                        );

                        swapchain_info.image_extent = extent;
                        swapchain_info.old_swapchain = swapchain.swapchain;

                        unsafe {
                            device.queue_wait_idle(queue)?;
                        }

                        swapchain = ash_abstractions::Swapchain::new(
                            &device,
                            &swapchain_loader,
                            swapchain_info,
                        )?;

                        unsafe {
                            device.reset_command_pool(
                                command_pool,
                                vk::CommandPoolResetFlags::empty(),
                            )?;

                            device.begin_command_buffer(
                                command_buffer,
                                &vk::CommandBufferBeginInfo::builder()
                                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                            )?;
                        }

                        depthbuffer.cleanup(&device, &mut allocator)?;

                        let mut init_resources = ash_abstractions::InitResources {
                            command_buffer,
                            device: &device,
                            allocator: &mut allocator,
                            debug_utils_loader: Some(&debug_utils_loader),
                        };

                        depthbuffer =
                            create_depthbuffer(extent.width, extent.height, &mut init_resources)?;

                        drop(init_resources);

                        unsafe {
                            device.end_command_buffer(command_buffer)?;
                            let fence =
                                device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

                            device.queue_submit(
                                queue,
                                &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
                                fence,
                            )?;

                            device.wait_for_fences(&[fence], true, u64::MAX)?;
                        }

                        swapchain_image_framebuffers = create_swapchain_image_framebuffers(
                            &device,
                            extent,
                            &swapchain,
                            &render_passes,
                            &depthbuffer,
                        )?;
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    let delta_time = 1.0 / 60.0;

                    let forwards = keyboard_state.forwards as i32 - keyboard_state.backwards as i32;
                    let right = keyboard_state.right as i32 - keyboard_state.left as i32;

                    let move_vec = camera.final_transform.rotation
                        * Vec3::new(right as f32, 0.0, -forwards as f32).clamp_length_max(1.0);

                    camera
                        .driver_mut::<dolly::drivers::Position>()
                        .translate(move_vec * delta_time * 400.0);

                    camera.update(delta_time);

                    push_constants.proj_view = {
                        let view = Mat4::look_at_rh(
                            camera.final_transform.position,
                            camera.final_transform.position + camera.final_transform.forward(),
                            camera.final_transform.up(),
                        );
                        perspective_matrix * view
                    };
                    push_constants.view_position = camera.final_transform.position.into();

                    window.request_redraw();
                }
                Event::RedrawRequested(_) => unsafe {
                    device.wait_for_fences(&[render_fence], true, u64::MAX)?;

                    device.reset_fences(&[render_fence])?;

                    device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

                    let swapchain_image_index = match swapchain_loader.acquire_next_image(
                        swapchain.swapchain,
                        u64::MAX,
                        present_semaphore,
                        vk::Fence::null(),
                    ) {
                        Ok((swapchain_image_index, _suboptimal)) => swapchain_image_index,
                        Err(error) => {
                            log::warn!("Next frame error: {:?}", error);
                            return Ok(());
                        }
                    };

                    let draw_framebuffer =
                        swapchain_image_framebuffers[swapchain_image_index as usize];

                    let draw_clear_values = [
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];

                    let area = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    };

                    let draw_render_pass_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(render_passes.draw)
                        .framebuffer(draw_framebuffer)
                        .render_area(area)
                        .clear_values(&draw_clear_values);

                    let viewport = *vk::Viewport::builder()
                        .x(0.0)
                        .y(0.0)
                        .width(extent.width as f32)
                        .height(extent.height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0);

                    {
                        device.begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )?;

                        device.cmd_begin_render_pass(
                            command_buffer,
                            &draw_render_pass_info,
                            vk::SubpassContents::INLINE,
                        );

                        device.cmd_set_scissor(command_buffer, 0, &[area]);
                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.grass,
                        );

                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.pipeline_layout,
                            0,
                            &[lights_ds],
                            &[],
                        );

                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[model_vertices.buffer],
                            &[0],
                        );

                        device.cmd_bind_index_buffer(
                            command_buffer,
                            model_indices.buffer,
                            0,
                            vk::IndexType::UINT32,
                        );

                        device.cmd_push_constants(
                            command_buffer,
                            pipelines.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytes_of(&push_constants),
                        );

                        device.cmd_draw_indexed(command_buffer, model_num_indices, 1, 0, 0, 0);

                        device.cmd_end_render_pass(command_buffer);

                        device.end_command_buffer(command_buffer)?;

                        device.queue_submit(
                            queue,
                            &[*vk::SubmitInfo::builder()
                                .wait_semaphores(&[present_semaphore])
                                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                                .command_buffers(&[command_buffer])
                                .signal_semaphores(&[render_semaphore])],
                            render_fence,
                        )?;

                        swapchain_loader.queue_present(
                            queue,
                            &vk::PresentInfoKHR::builder()
                                .wait_semaphores(&[render_semaphore])
                                .swapchains(&[swapchain.swapchain])
                                .image_indices(&[swapchain_image_index]),
                        )?;
                    }
                },
                Event::LoopDestroyed => {
                    unsafe {
                        device.queue_wait_idle(queue)?;
                    }

                    {
                        depthbuffer.cleanup(&device, &mut allocator)?;
                        model_vertices.cleanup(&device, &mut allocator)?;
                        model_indices.cleanup(&device, &mut allocator)?;
                        lights_buffer.cleanup(&device, &mut allocator)?;
                        tonemapping_params_buffer.cleanup(&device, &mut allocator)?;
                    }
                }
                _ => {}
            }

            Ok(())
        };

        if let Err(loop_closure) = loop_closure() {
            log::error!("Error: {}", loop_closure);
        }
    });
}

fn create_swapchain_image_framebuffers(
    device: &ash::Device,
    extent: vk::Extent2D,
    swapchain: &ash_abstractions::Swapchain,
    render_passes: &RenderPasses,
    depthbuffer: &ash_abstractions::Image,
) -> anyhow::Result<Vec<vk::Framebuffer>> {
    swapchain
        .image_views
        .iter()
        .map(|image_view| {
            unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(render_passes.draw)
                        .attachments(&[*image_view, depthbuffer.view])
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(|err| err.into())
        })
        .collect()
}

#[derive(Default)]
struct KeyboardState {
    forwards: bool,
    right: bool,
    backwards: bool,
    left: bool,
}

struct Pipelines {
    grass: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl Pipelines {
    fn new(
        device: &ash::Device,
        render_passes: &RenderPasses,
        lights_dsl: vk::DescriptorSetLayout,
        pipeline_cache: vk::PipelineCache,
    ) -> anyhow::Result<Self> {
        let vertex_entry_point = CString::new("vertex")?;
        let fragment_entry_point = CString::new("fragment")?;

        let module =
            ash_abstractions::load_shader_module(include_bytes!("../shader.spv"), &device)?;

        let vertex_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_entry_point);

        let fragment_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_entry_point);

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[lights_dsl])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
                None,
            )
        }?;

        let stages = &[*vertex_stage, *fragment_stage];

        let graphics_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
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
            vertex_attributes: &ash_abstractions::create_vertex_attribute_descriptions(
                0,
                &[
                    ash_abstractions::VertexAttribute::Vec3,
                    ash_abstractions::VertexAttribute::Vec3,
                    ash_abstractions::VertexAttribute::Vec2,
                    ash_abstractions::VertexAttribute::Uint,
                ],
            ),
            vertex_bindings: &[*vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(std::mem::size_of::<Vertex>() as u32)],
            colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)],
        };

        let baked = graphics_pipeline_desc.as_baked();

        let graphics_pipeline_desc =
            baked.as_pipeline_create_info(stages, pipeline_layout, render_passes.draw, 0);

        let pipelines = unsafe {
            device.create_graphics_pipelines(pipeline_cache, &[*graphics_pipeline_desc], None)
        }
        .map_err(|(_, err)| err)?;

        Ok(Self {
            grass: pipelines[0],
            pipeline_layout,
        })
    }
}

struct Vertex {
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    material: u32,
}

struct RenderPasses {
    draw: vk::RenderPass,
    tonemap: vk::RenderPass,
}

impl RenderPasses {
    fn new(device: &ash::Device, surface_format: vk::Format) -> anyhow::Result<Self> {
        let draw_attachments = [
            // HDR framebuffer
            *vk::AttachmentDescription::builder()
                .format(surface_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];

        let hdr_framebuffer_ref = [*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let depth_attachment_ref = *vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let draw_subpass = [*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&hdr_framebuffer_ref)
            .depth_stencil_attachment(&depth_attachment_ref)];

        let draw_subpass_dependency = [*vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let draw_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&draw_attachments)
                    .subpasses(&draw_subpass)
                    .dependencies(&draw_subpass_dependency),
                None,
            )
        }?;

        let tonemap_attachments = [
            // Swapchain framebuffer
            *vk::AttachmentDescription::builder()
                .format(surface_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];

        let swapchain_framebuffer_ref = [*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let tonemap_subpass = [*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&swapchain_framebuffer_ref)];

        let tonemap_subpass_dependency = [*vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let tonemap_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&tonemap_attachments)
                    .subpasses(&tonemap_subpass)
                    .dependencies(&tonemap_subpass_dependency),
                None,
            )
        }?;

        Ok(Self {
            draw: draw_render_pass,
            tonemap: tonemap_render_pass,
        })
    }
}

fn create_depthbuffer(
    width: u32,
    height: u32,
    init_resources: &mut ash_abstractions::InitResources,
) -> anyhow::Result<ash_abstractions::Image> {
    ash_abstractions::Image::new(
        &ash_abstractions::ImageDescriptor {
            width,
            height,
            name: "depthbuffer",
            mip_levels: 1,
            format: vk::Format::D32_SFLOAT,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            next_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
            next_layout: vk_sync::ImageLayout::Optimal,
        },
        init_resources,
    )
}

fn load_gltf_from_bytes(
    bytes: &[u8],
    init_resources: &mut ash_abstractions::InitResources,
) -> anyhow::Result<(ash_abstractions::Buffer, ash_abstractions::Buffer, u32)> {
    let gltf = gltf::Gltf::from_slice(bytes)?;

    let buffer_blob = gltf.blob.as_ref().unwrap();

    let mut indices = Vec::new();
    let mut vertices = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let material_id = primitive.material().index().unwrap_or(0);

            let reader = primitive.reader(|_| Some(buffer_blob));

            let read_indices = reader.read_indices().unwrap().into_u32();

            let num_vertices = vertices.len() as u32;

            indices.extend(read_indices.map(|index| index + num_vertices));

            let positions = reader.read_positions().unwrap();
            let normals = reader.read_normals().unwrap();
            let uvs = reader.read_tex_coords(0).unwrap().into_f32();

            for ((position, normal), uv) in positions.zip(normals).zip(uvs) {
                vertices.push(Vertex {
                    position: position.into(),
                    uv: uv.into(),
                    normal: normal.into(),
                    material: material_id as u32,
                });
            }
        }
    }

    /*
    let material = gltf.materials().next().unwrap();

    let texture = material.emissive_texture().unwrap();

    let texture = load_texture_from_gltf(
        texture.texture(),
        "emissive texture",
        buffer_blob,
        init_resources,
        buffers_to_cleanup,
    )?;
    */

    let num_indices = indices.len() as u32;

    let vertices = ash_abstractions::Buffer::new(
        unsafe { cast_slice(&vertices) },
        "vertices",
        vk::BufferUsageFlags::VERTEX_BUFFER,
        init_resources,
    )?;

    let indices = ash_abstractions::Buffer::new(
        unsafe { cast_slice(&indices) },
        "indices",
        vk::BufferUsageFlags::INDEX_BUFFER,
        init_resources,
    )?;

    Ok((vertices, indices, num_indices))
}

unsafe fn cast_slice<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        slice as *const [T] as *const u8,
        slice.len() * std::mem::size_of::<T>(),
    )
}

unsafe fn bytes_of<T>(reference: &T) -> &[u8] {
    std::slice::from_raw_parts(reference as *const T as *const u8, std::mem::size_of::<T>())
}

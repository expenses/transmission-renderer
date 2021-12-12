use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::vk;
use crate::GBuffer;

pub struct RenderPasses {
    pub(crate) draw_forwards: vk::RenderPass,
    pub(crate) draw_deferred: vk::RenderPass,
    pub(crate) transmission: vk::RenderPass,
    pub(crate) tonemap: vk::RenderPass,
    pub(crate) sun_shadow: vk::RenderPass,
    pub(crate) defer: vk::RenderPass,
}

impl RenderPasses {
    pub fn new(
        device: &ash::Device,
        debug_utils_loader: &DebugUtilsLoader,
        surface_format: vk::Format,
    ) -> anyhow::Result<Self> {
        let depth_attachment_ref = *vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let single_attachment_ref = [*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let draw_attachments = |is_deferred| {
            [
                // Depth buffer
                *vk::AttachmentDescription::builder()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(if is_deferred { vk::AttachmentLoadOp::LOAD } else { vk::AttachmentLoadOp::CLEAR })
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                // HDR framebuffers
                *vk::AttachmentDescription::builder()
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    // No initial layout as the framebuffer is either `COLOR_ATTACHMENT_OPTIMAL` (first frame)
                    // or `SHADER_READ_ONLY_OPTIMAL` (all the other frames, as the output from the transmission pass.)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                *vk::AttachmentDescription::builder()
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    //.initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            ]
        };

        let draw_deferred_attachments = draw_attachments(true);
        let draw_forwards_attachments = draw_attachments(false);

        let hdr_framebuffer_refs = [
            *vk::AttachmentReference::builder()
                .attachment(1)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            *vk::AttachmentReference::builder()
                .attachment(2)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];

        let draw_deferred_subpasses = [
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&hdr_framebuffer_refs),
        ];

        let draw_deferred_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&draw_deferred_attachments)
                    .subpasses(&draw_deferred_subpasses),
                None,
            )
        }?;

        let draw_forwards_subpasses = [
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .depth_stencil_attachment(&depth_attachment_ref),
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .depth_stencil_attachment(&depth_attachment_ref)
                .color_attachments(&hdr_framebuffer_refs),
        ];

        // https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
        // says I can ignore external subpass dependencies *.
        //
        // * unless they need to transition layouts!
        let draw_forwards_subpass_dependencies = [
            *vk::SubpassDependency::builder()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                .dst_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            *vk::SubpassDependency::builder()
                .src_subpass(1)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];

        let draw_forwards_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&draw_forwards_attachments)
                    .subpasses(&draw_forwards_subpasses)
                    .dependencies(&draw_forwards_subpass_dependencies),
                None,
            )
        }?;

        let tonemap_attachments = [
            // swapchain image
            *vk::AttachmentDescription::builder()
                .format(surface_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        ];


        let single_subpass = [
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&single_attachment_ref),
        ];

        let tonemap_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&tonemap_attachments)
                    .subpasses(&single_subpass),
                None,
            )
        }?;

        let transmission_attachments = [
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            // HDR framebuffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];

        let hdr_framebuffer_ref = [*vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let transmission_subpass = [*vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_attachment_ref),*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&hdr_framebuffer_ref)
            .depth_stencil_attachment(&depth_attachment_ref)];

        let transmission_subpass_dependency = [
            *vk::SubpassDependency::builder()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ),
            *vk::SubpassDependency::builder()
                .src_subpass(1)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];

        let transmission_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&transmission_attachments)
                    .subpasses(&transmission_subpass)
                    .dependencies(&transmission_subpass_dependency),
                None,
            )
        }?;

        let sun_shadow_attachment = [
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            // Sun buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];

        let sun_shadow_buffer_ref = [*vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let sun_shadow_subpass = [
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&sun_shadow_buffer_ref)
                .depth_stencil_attachment(&depth_attachment_ref),
        ];

        let sun_shadow_subpass_dependencies = [
            *vk::SubpassDependency::builder()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];

        let sun_shadow_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&sun_shadow_attachment)
                    .subpasses(&sun_shadow_subpass)
                    .dependencies(&sun_shadow_subpass_dependencies),
                None,
            )
        }?;

        let defer_attachments = [
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            // G Buffer
            *vk::AttachmentDescription::builder()
                .format(GBuffer::POSITION_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            *vk::AttachmentDescription::builder()
                .format(GBuffer::NORMAL_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            *vk::AttachmentDescription::builder()
                .format(GBuffer::UV_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            *vk::AttachmentDescription::builder()
                .format(GBuffer::MATERIAL_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];

        let defer_attachment_refs = [*vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            *vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            *vk::AttachmentReference::builder()
            .attachment(3)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            *vk::AttachmentReference::builder()
            .attachment(4)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let defer_subpass = [
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&defer_attachment_refs)
                .depth_stencil_attachment(&depth_attachment_ref),
        ];

        let defer_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&defer_attachments)
                    .subpasses(&defer_subpass),
                None,
            )
        }?;

        let set_name = |render_pass, name| {
            ash_abstractions::set_object_name(
                device,
                debug_utils_loader,
                render_pass,
                name
            )
        };

        set_name(draw_deferred_render_pass, "draw deferred render pass")?;
        set_name(tonemap_render_pass, "tonemap render pass")?;
        set_name(transmission_render_pass, "transmission render pass")?;
        set_name(sun_shadow_render_pass, "sun shadow render pass")?;
        set_name(defer_render_pass, "defer render pass")?;

        Ok(Self {
            draw_forwards: draw_forwards_render_pass,
            draw_deferred: draw_deferred_render_pass,
            tonemap: tonemap_render_pass,
            transmission: transmission_render_pass,
            sun_shadow: sun_shadow_render_pass,
            defer: defer_render_pass,
        })
    }
}

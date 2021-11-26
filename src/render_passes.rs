use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::vk;

#[derive(Clone, Copy)]
pub struct RenderPasses {
    pub draw: vk::RenderPass,
    pub transmission: vk::RenderPass,
    pub tonemap: vk::RenderPass,
}

impl RenderPasses {
    pub fn new(
        device: &ash::Device,
        debug_utils_loader: &DebugUtilsLoader,
        surface_format: vk::Format,
    ) -> anyhow::Result<Self> {
        // todo: We get some syncronisation validation warnings by not specifying initial layouts here.
        let draw_attachments = [
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
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
        ];

        let depth_attachment_ref = *vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let hdr_framebuffer_refs = [
            *vk::AttachmentReference::builder()
                .attachment(1)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            *vk::AttachmentReference::builder()
                .attachment(2)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];

        let draw_subpasses = [
            // Depth pre-pass
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .depth_stencil_attachment(&depth_attachment_ref),
            // Colour pass
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&hdr_framebuffer_refs)
                .depth_stencil_attachment(&depth_attachment_ref),
            // Second depth pre-pass for transmissive objects.
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .depth_stencil_attachment(&depth_attachment_ref),
        ];

        // https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
        // says I can ignore external subpass dependencies.
        let draw_subpass_dependencices = [
            *vk::SubpassDependency::builder()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            *vk::SubpassDependency::builder()
                .src_subpass(1)
                .dst_subpass(2)
                .src_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                .dst_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
        ];

        let draw_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&draw_attachments)
                    .subpasses(&draw_subpasses)
                    .dependencies(&draw_subpass_dependencices),
                None,
            )
        }?;

        let tonemap_attachments = [
            // swapchain image
            *vk::AttachmentDescription::builder()
                .format(surface_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        ];

        let swapchain_image_ref = [*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let tonemap_subpasses = [
            // Tonemapping pass
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&swapchain_image_ref),
        ];

        let tonemap_subpass_dependencies = [];

        let tonemap_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&tonemap_attachments)
                    .subpasses(&tonemap_subpasses)
                    .dependencies(&tonemap_subpass_dependencies),
                None,
            )
        }?;

        let transmission_attachments = [
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            // HDR framebuffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];

        let depth_attachment_ref = *vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let hdr_framebuffer_ref = [*vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let transmission_subpass = [*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&hdr_framebuffer_ref)
            .depth_stencil_attachment(&depth_attachment_ref)];

        let transmission_subpass_dependency = [];

        let transmission_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&transmission_attachments)
                    .subpasses(&transmission_subpass)
                    .dependencies(&transmission_subpass_dependency),
                None,
            )
        }?;

        ash_abstractions::set_object_name(
            device,
            debug_utils_loader,
            draw_render_pass,
            "draw render pass",
        )?;
        ash_abstractions::set_object_name(
            device,
            debug_utils_loader,
            tonemap_render_pass,
            "tonemap render pass",
        )?;
        ash_abstractions::set_object_name(
            device,
            debug_utils_loader,
            transmission_render_pass,
            "transmission render pass",
        )?;

        Ok(Self {
            draw: draw_render_pass,
            tonemap: tonemap_render_pass,
            transmission: transmission_render_pass,
        })
    }
}

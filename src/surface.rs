use crate::QueueFamilyIndices;
use crate::QueueWrapper;
use crate::SwapChainSupportDetails;
use crate::SwapchainCreateInfoKHR;
use ash::extensions::khr::Surface;
use ash::extensions::khr::Swapchain;
use ash::vk::AccessFlags;
use ash::vk::AttachmentDescription;
use ash::vk::AttachmentLoadOp;
use ash::vk::AttachmentReference;
use ash::vk::AttachmentStoreOp;
use ash::vk::ComponentMapping;
use ash::vk::ComponentSwizzle;
use ash::vk::Fence;
use ash::vk::FramebufferCreateInfo;
use ash::vk::ImageAspectFlags;
use ash::vk::ImageLayout;
use ash::vk::ImageSubresourceRange;
use ash::vk::ImageViewCreateInfo;
use ash::vk::ImageViewType;
use ash::vk::PipelineBindPoint;
use ash::vk::PipelineStageFlags;
use ash::vk::PresentInfoKHR;
use ash::vk::RenderPass;
use ash::vk::RenderPassCreateInfo;
use ash::vk::SampleCountFlags;
use ash::vk::Semaphore;
use ash::vk::SubpassDependency;
use ash::vk::SubpassDescription;
use ash::vk::{
    self, ColorSpaceKHR, CompositeAlphaFlagsKHR, Extent2D, Format, Framebuffer, Image,
    ImageUsageFlags, ImageView, PresentModeKHR, SharingMode, SurfaceCapabilitiesKHR,
    SurfaceFormatKHR, SwapchainKHR,
};
use ash::Device;
use ash::{vk::SurfaceKHR, Entry, Instance};
use winit::dpi::PhysicalSize;
#[cfg(target_os = "macos")]
use winit::raw_window_handle::AppKitWindowHandle;
use winit::raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle,
    WaylandWindowHandle,
};
use winit::window::Window;

pub struct WindowSurface {
    surface_fns: Surface,
    pub handle: SurfaceKHR,
}

impl WindowSurface {
    pub fn new(entry: &Entry, instance: &Instance, window: &Window) -> Self {
        let window_handle = window
            .window_handle()
            .expect("failed to get window handle")
            .as_raw();
        let display_handle = window
            .display_handle()
            .expect("failed to get display handle")
            .as_raw();

        let surface = match (window_handle, display_handle) {
            (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
                Self::create_wayland_surface(&entry, &instance, window, display)
            }
            #[cfg(target_os = "macos")]
            (RawWindowHandle::AppKit(window), RawDisplayHandle::AppKit(_)) => {
                Self::create_app_kit_surface(&entry, &instance, window)
            }

            _ => panic!("unsupported windowing system"),
        };

        Self {
            surface_fns: Surface::new(entry, instance),
            handle: surface,
        }
    }

    #[cfg(target_os = "macos")]
    fn create_app_kit_surface(
        entry: &Entry,
        instance: &Instance,
        window_handle: AppKitWindowHandle,
    ) -> SurfaceKHR {
        use ash::extensions::ext::MetalSurface;
        use raw_window_metal::{appkit::metal_layer_from_handle, Layer::*};
        let surface = MetalSurface::new(&entry, &instance);

        let layer = match unsafe { metal_layer_from_handle(window_handle) } {
            Existing(layer) | Allocated(layer) => layer,
        };

        let create_info = MetalSurfaceCreateInfoEXT::builder().layer(layer.cast());

        unsafe { surface.create_metal_surface(&create_info, None) }
            .expect("failed to create wayland surface")
    }

    fn create_wayland_surface(
        entry: &Entry,
        instance: &Instance,
        window_handle: WaylandWindowHandle,
        display_handle: WaylandDisplayHandle,
    ) -> SurfaceKHR {
        use ash::extensions::khr::WaylandSurface;
        use ash::vk::WaylandSurfaceCreateInfoKHR;

        let surface = WaylandSurface::new(&entry, &instance);

        let create_info = WaylandSurfaceCreateInfoKHR {
            display: display_handle.display.as_ptr(),
            surface: window_handle.surface.as_ptr(),
            ..Default::default()
        };

        unsafe { surface.create_wayland_surface(&create_info, None) }
            .expect("failed to create wayland surface")
    }

    pub unsafe fn cleanup_surface(&mut self) {
        self.surface_fns.destroy_surface(self.handle, None);
    }
}

pub struct SwapChain {
    swapchain_fns: Swapchain,
    pub data: SwapChainData,
    pub render_pass: RenderPass,
    image_views: Vec<ImageView>,
    pub framebuffers: Vec<Framebuffer>,
}

impl SwapChain {
    pub(crate) fn new(
        instance: &Instance,
        device: &Device,
        window: &Window,
        surface: SurfaceKHR,
        swap_chain_support: &SwapChainSupportDetails,
        queue_family_indices: &QueueFamilyIndices,
    ) -> Self {
        let swapchain_fns = Swapchain::new(instance, device);
        let data = SwapChainData::new(
            &swapchain_fns,
            surface,
            &swap_chain_support,
            window.inner_size(),
            queue_family_indices.graphics,
            queue_family_indices.present,
        );

        let image_views = Self::create_image_views(device, &data);
        let render_pass = Self::create_render_pass(&device, &data);
        let framebuffers = Self::create_framebuffers(&device, &data, &image_views, render_pass);

        Self {
            swapchain_fns,
            data,
            image_views,
            render_pass,
            framebuffers,
        }
    }

    fn create_image_views(device: &Device, swap_chain_data: &SwapChainData) -> Vec<ImageView> {
        swap_chain_data
            .images
            .iter()
            .map(|&img| {
                let create_info = ImageViewCreateInfo {
                    image: img,
                    view_type: ImageViewType::TYPE_2D,
                    format: swap_chain_data.surface_format.format,

                    components: ComponentMapping {
                        r: ComponentSwizzle::IDENTITY,
                        g: ComponentSwizzle::IDENTITY,
                        b: ComponentSwizzle::IDENTITY,
                        a: ComponentSwizzle::IDENTITY,
                    },

                    subresource_range: ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },

                    ..Default::default()
                };

                unsafe { device.create_image_view(&create_info, None) }
                    .expect("failed to create image view")
            })
            .collect()
    }

    fn create_render_pass(device: &Device, swap_chain_data: &SwapChainData) -> RenderPass {
        let color_attachments = [AttachmentDescription {
            format: swap_chain_data.surface_format.format,
            samples: SampleCountFlags::TYPE_1,

            load_op: AttachmentLoadOp::CLEAR,
            store_op: AttachmentStoreOp::STORE,

            stencil_load_op: AttachmentLoadOp::DONT_CARE,
            stencil_store_op: AttachmentStoreOp::DONT_CARE,

            initial_layout: ImageLayout::UNDEFINED,
            final_layout: ImageLayout::PRESENT_SRC_KHR,

            ..Default::default()
        }];

        let color_attachment_refs = [AttachmentReference {
            attachment: 0,
            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpasses = [SubpassDescription {
            pipeline_bind_point: PipelineBindPoint::GRAPHICS,
            color_attachment_count: color_attachment_refs.len() as u32,
            p_color_attachments: color_attachment_refs.as_ptr(),

            ..Default::default()
        }];

        let dependencies = [SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,

            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: AccessFlags::empty(),

            dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,

            ..Default::default()
        }];

        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(&color_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        unsafe { device.create_render_pass(&render_pass_info, None) }
            .expect("failed to create render pass!")
    }

    fn create_framebuffers(
        device: &Device,
        swap_chain_data: &SwapChainData,
        image_views: &Vec<ImageView>,
        render_pass: RenderPass,
    ) -> Vec<Framebuffer> {
        image_views
            .iter()
            .map(|&img_view| {
                let attachments = [img_view];

                let framebuffer_info = FramebufferCreateInfo {
                    render_pass,
                    attachment_count: attachments.len() as u32,
                    p_attachments: attachments.as_ptr(),

                    width: swap_chain_data.extent.width,
                    height: swap_chain_data.extent.height,
                    layers: 1,

                    ..Default::default()
                };

                unsafe { device.create_framebuffer(&framebuffer_info, None) }
                    .expect("failed to create framebuffer!")
            })
            .collect()
    }

    pub unsafe fn cleanup_swap_chain(&mut self, device: &Device) {
        self.framebuffers
            .iter()
            .for_each(|&fbuf| device.destroy_framebuffer(fbuf, None));

        device.destroy_render_pass(self.render_pass, None);

        self.image_views
            .iter()
            .for_each(|&img_view| device.destroy_image_view(img_view, None));

        self.swapchain_fns.destroy_swapchain(self.data.chain, None);
    }

    pub fn recreate_swap_chain(
        &mut self,
        device: &Device,
        surface: SurfaceKHR,
        window_size: Option<PhysicalSize<u32>>,
    ) {
        unsafe {
            device
                .device_wait_idle()
                .expect("failed while waiting for device idle")
        };

        unsafe { self.cleanup_swap_chain(device) };

        self.data
            .recreate_swap_chain(&self.swapchain_fns, surface, window_size);
        self.image_views = Self::create_image_views(device, &self.data);
        self.render_pass = Self::create_render_pass(device, &self.data);
        self.framebuffers =
            Self::create_framebuffers(device, &self.data, &self.image_views, self.render_pass);
    }

    pub fn acquire_next_image(&self, img_available: Semaphore) -> Result<u32, vk::Result> {
        match unsafe {
            self.swapchain_fns.acquire_next_image(
                self.data.chain,
                u64::MAX,
                img_available,
                Fence::null(),
            )
        } {
            Ok((index, _)) => Ok(index),
            Err(err @ vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(err),

            _ => panic!("failed to acquire next image from swap chain"),
        }
    }

    pub fn present(
        &self,
        present_queue: &QueueWrapper,
        present_info: &PresentInfoKHR,
    ) -> Result<bool, vk::Result> {
        unsafe {
            self.swapchain_fns
                .queue_present(present_queue.handle, &present_info)
        }
    }
}

pub struct SwapChainData {
    pub chain: SwapchainKHR,
    images: Vec<Image>,
    pub extent: Extent2D,
    present_mode: PresentModeKHR,
    surface_format: SurfaceFormatKHR,
    surface_capabilities: SurfaceCapabilitiesKHR,
    queue_indices: [u32; 2],
}

impl SwapChainData {
    fn new(
        swapchain_fns: &Swapchain,
        surface: SurfaceKHR,
        swap_chain_support: &SwapChainSupportDetails,
        window_size: PhysicalSize<u32>,
        graphics_queue_index: u32,
        present_queue_index: u32,
    ) -> SwapChainData {
        let surface_format = Self::choose_swap_surface_format(&swap_chain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swap_chain_support.present_modes);
        let extent = Self::choose_swap_extent(&swap_chain_support.capabilities, window_size);
        let queue_indices = [graphics_queue_index, present_queue_index];

        let (swap_chain, images) = Self::create_swapchain(
            swapchain_fns,
            surface,
            &surface_format,
            &swap_chain_support.capabilities,
            extent,
            present_mode,
            &queue_indices,
        );
        SwapChainData {
            chain: swap_chain,
            extent,
            images,
            present_mode,
            surface_format,
            surface_capabilities: swap_chain_support.capabilities,
            queue_indices,
        }
    }

    fn create_swapchain(
        swapchain_fns: &Swapchain,
        surface: SurfaceKHR,
        surface_format: &SurfaceFormatKHR,
        capabilities: &SurfaceCapabilitiesKHR,
        extent: Extent2D,
        present_mode: PresentModeKHR,
        queue_indices: &[u32; 2],
    ) -> (SwapchainKHR, Vec<Image>) {
        let image_count = (capabilities.min_image_count + 1).clamp(
            capabilities.min_image_count,
            if capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                capabilities.max_image_count
            },
        );

        let queue_indices: &[u32] = if queue_indices[0] == queue_indices[1] {
            &[]
        } else {
            queue_indices
        };

        let create_info = SwapchainCreateInfoKHR {
            surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,

            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,

            image_sharing_mode: if queue_indices.is_empty() {
                SharingMode::EXCLUSIVE
            } else {
                SharingMode::CONCURRENT
            },
            queue_family_index_count: queue_indices.len() as u32,
            p_queue_family_indices: queue_indices.as_ptr(),

            pre_transform: capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: SwapchainKHR::null(),
            ..Default::default()
        };

        let swap_chain = unsafe { swapchain_fns.create_swapchain(&create_info, None) }
            .expect("failed to create swapchain");

        let images = unsafe { swapchain_fns.get_swapchain_images(swap_chain) }
            .expect("failed to get swapchain images");

        (swap_chain, images)
    }

    fn recreate_swap_chain(
        &mut self,
        swapchain_fns: &Swapchain,
        surface: SurfaceKHR,
        window_size: Option<PhysicalSize<u32>>,
    ) {
        if let Some(window_size) = window_size {
            self.extent = Self::choose_swap_extent(&self.surface_capabilities, window_size);
        }
        let (chain, images) = Self::create_swapchain(
            swapchain_fns,
            surface,
            &self.surface_format,
            &self.surface_capabilities,
            self.extent,
            self.present_mode,
            &self.queue_indices,
        );

        self.chain = chain;
        self.images = images;
    }

    fn choose_swap_surface_format(available_formats: &Vec<SurfaceFormatKHR>) -> SurfaceFormatKHR {
        *available_formats
            .iter()
            .find(|format| {
                format.format == Format::B8G8R8A8_SRGB
                    && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    fn choose_swap_present_mode(available_modes: &Vec<PresentModeKHR>) -> PresentModeKHR {
        *available_modes
            .iter()
            .find(|&&mode| mode == PresentModeKHR::MAILBOX)
            .unwrap_or(&PresentModeKHR::FIFO)
    }

    fn choose_swap_extent(
        capabilities: &SurfaceCapabilitiesKHR,
        window_size: PhysicalSize<u32>,
    ) -> Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        } else {
            Extent2D {
                width: window_size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }
}

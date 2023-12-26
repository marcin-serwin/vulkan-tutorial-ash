use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
#[cfg(debug_assertions)]
use std::mem::ManuallyDrop;

use ash::extensions::ext::DebugUtils;
use ash::{extensions::khr::*, vk::*, Device, Entry, Instance};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle,
    WaylandWindowHandle,
};
use winit::window::Window;

macro_rules! include_shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name))
    };
}

macro_rules! cstr {
    ($str:literal) => {{
        const RESULT: &CStr =
            unsafe { CStr::from_bytes_with_nul_unchecked(concat!($str, "\0").as_bytes()) };
        RESULT
    }};
}

unsafe fn name_to_cstr(name: &[c_char]) -> &CStr {
    CStr::from_bytes_until_nul(std::mem::transmute(name)).unwrap_unchecked()
}

const DEVICE_EXTENSIONS: [&CStr; 1] = [KhrSwapchainFn::name()];

#[cfg(not(debug_assertions))]
const LAYERS: [&CStr; 0] = [];
#[cfg(debug_assertions)]
const LAYERS: [&CStr; 1] = [cstr!("VK_LAYER_KHRONOS_validation")];
fn get_validation_layers(entry: &Entry) -> [&'static CStr; LAYERS.len()] {
    if LAYERS.is_empty() {
        return LAYERS;
    }
    let supported_layers = entry.enumerate_instance_layer_properties().unwrap();

    let names: Vec<&CStr> = supported_layers
        .iter()
        .map(|layer| unsafe { name_to_cstr(&layer.layer_name) })
        .collect();

    if LAYERS.iter().any(|ext| !names.contains(ext)) {
        panic!("Required validation layer not found");
    }

    LAYERS
}

#[cfg(all(target_os = "linux", not(debug_assertions)))]
const EXTENSIONS: [&CStr; 2] = [KhrSurfaceFn::name(), KhrWaylandSurfaceFn::name()];
#[cfg(all(target_os = "macos", not(debug_assertions)))]
const EXTENSIONS: [&CStr; 1] = [KhrPortabilityEnumerationFn::name()];
#[cfg(all(target_os = "linux", debug_assertions))]
const EXTENSIONS: [&CStr; 3] = [
    KhrSurfaceFn::name(),
    KhrWaylandSurfaceFn::name(),
    ExtDebugUtilsFn::name(),
];
#[cfg(all(target_os = "macos", debug_assertions))]
const EXTENSIONS: [&CStr; 2] = [KhrPortabilityEnumerationFn::name(), ExtDebugUtilsFn::name()];

fn get_extensions(entry: &Entry) -> [&'static CStr; EXTENSIONS.len()] {
    if EXTENSIONS.is_empty() {
        return EXTENSIONS;
    }
    let supported_extensions = entry.enumerate_instance_extension_properties(None).unwrap();
    let names: Vec<&CStr> = supported_extensions
        .iter()
        .map(|ext| unsafe { name_to_cstr(&ext.extension_name) })
        .collect();

    if EXTENSIONS.iter().any(|ext| !names.contains(ext)) {
        panic!("Required extension not found");
    }

    EXTENSIONS
}

struct Messenger {
    debug_utils: DebugUtils,
    messenger: DebugUtilsMessengerEXT,
}

impl Messenger {
    unsafe extern "system" fn callback(
        message_severity: DebugUtilsMessageSeverityFlagsEXT,
        _message_type: DebugUtilsMessageTypeFlagsEXT,
        cb_data: *const DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut c_void,
    ) -> u32 {
        let message = CStr::from_ptr((*cb_data).p_message)
            .to_str()
            .expect("invalid message received from validation");
        eprintln!("{message_severity:?}: {message}");
        if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            panic!("Validation error");
        }
        FALSE
    }

    unsafe fn new(entry: &Entry, instance: &Instance) -> Self {
        let debug_utils = DebugUtils::new(entry, instance);
        let create_info = DebugUtilsMessengerCreateInfoEXT {
            message_severity: DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | DebugUtilsMessageSeverityFlagsEXT::INFO
                | DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::ERROR,

            message_type: DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,

            pfn_user_callback: Some(Messenger::callback),
            ..Default::default()
        };

        let messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .unwrap()
        };

        Self {
            debug_utils,
            messenger,
        }
    }
}

impl Drop for Messenger {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

#[derive(Debug, Default)]
struct SwapChainSupportDetails {
    capabilities: SurfaceCapabilitiesKHR,
    formats: Vec<SurfaceFormatKHR>,
    present_modes: Vec<PresentModeKHR>,
}

#[derive(Debug, Default)]
struct PartialQueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl PartialQueueFamilyIndices {
    fn is_total(&self) -> bool {
        self.as_total().is_some()
    }

    fn as_total(&self) -> Option<QueueFamilyIndices> {
        Some(QueueFamilyIndices {
            graphics_family: self.graphics_family?,
            present_family: self.present_family?,
        })
    }
}

#[derive(Debug, Default)]
struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

struct QueueFamily {
    graphics_queue: Queue,
    present_queue: Queue,
}

struct SwapChainData {
    swap_chain: SwapchainKHR,
    images: Vec<Image>,
    format: Format,
    extent: Extent2D,
}

struct HelloTriangleApplication {
    entry: Entry,
    instance: Instance,
    surface: SurfaceKHR,
    device: Device,

    queue_family: QueueFamily,
    swap_chain_data: SwapChainData,
    image_views: Vec<ImageView>,

    render_pass: RenderPass,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,

    framebuffers: Vec<Framebuffer>,

    #[cfg(debug_assertions)]
    messenger: ManuallyDrop<Messenger>,
}

impl HelloTriangleApplication {
    fn new(entry: Entry, window: &Window) -> Self {
        let instance = Self::create_instance(&entry);
        #[cfg(debug_assertions)]
        let messenger = ManuallyDrop::new(unsafe { Messenger::new(&entry, &instance) });
        let surface = Self::create_surface(&entry, &instance, window);
        let (device, queue_indices, swap_chain_support_details) =
            Self::create_logical_device(&entry, &instance, surface);
        let queue_family = QueueFamily {
            graphics_queue: unsafe { device.get_device_queue(queue_indices.graphics_family, 0) },
            present_queue: unsafe { device.get_device_queue(queue_indices.present_family, 0) },
        };

        let swap_chain_data = Self::create_swap_chain(
            &Swapchain::new(&instance, &device),
            window,
            surface,
            swap_chain_support_details,
            queue_indices,
        );
        let image_views = Self::create_image_views(&device, &swap_chain_data);

        let render_pass = Self::create_render_pass(&device, &swap_chain_data);
        let (pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(&device, &swap_chain_data, render_pass);

        let framebuffers =
            Self::create_framebuffers(&device, &swap_chain_data, &image_views, render_pass);

        Self {
            entry,
            instance,
            surface,
            device,

            queue_family,
            swap_chain_data,
            image_views,

            render_pass,

            pipeline_layout,
            pipeline,

            framebuffers,

            #[cfg(debug_assertions)]
            messenger,
        }
    }

    fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> SurfaceKHR {
        let window_handle = window
            .window_handle()
            .expect("failed to get window handle")
            .as_raw();
        let display_handle = window
            .display_handle()
            .expect("failed to get display handle")
            .as_raw();

        match (window_handle, display_handle) {
            (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
                Self::create_wayland_surface(&entry, &instance, &window, &display)
            }

            _ => panic!("unsupported windowing system"),
        }
    }

    fn create_wayland_surface(
        entry: &Entry,
        instance: &Instance,
        window_handle: &WaylandWindowHandle,
        display_handle: &WaylandDisplayHandle,
    ) -> SurfaceKHR {
        let surface = WaylandSurface::new(&entry, &instance);

        let create_info = WaylandSurfaceCreateInfoKHR {
            display: display_handle.display.as_ptr(),
            surface: window_handle.surface.as_ptr(),
            ..Default::default()
        };

        let surface = unsafe { surface.create_wayland_surface(&create_info, None) }
            .expect("failed to create wayland surface");
        surface
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_info = ApplicationInfo {
            api_version: make_api_version(0, 1, 0, 0),
            p_application_name: cstr!("MyTest").as_ptr(),

            ..Default::default()
        };

        let extensions = get_extensions(entry);
        let validation_layers = get_validation_layers(entry);

        #[cfg(debug_assertions)]
        let debug_create_info = DebugUtilsMessengerCreateInfoEXT {
            message_severity: DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | DebugUtilsMessageSeverityFlagsEXT::INFO
                | DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::ERROR,

            message_type: DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,

            pfn_user_callback: Some(Messenger::callback),
            ..Default::default()
        };

        let create_info = InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: extensions.map(CStr::as_ptr).as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_layer_names: validation_layers.map(CStr::as_ptr).as_ptr(),
            enabled_layer_count: validation_layers.len() as u32,
            #[cfg(debug_assertions)]
            p_next: &debug_create_info as *const _ as *const c_void,
            #[cfg(target_os = "macos")]
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
            ..Default::default()
        };

        // let exts = entry.enumerate_instance_extension_properties(None).unwrap();

        unsafe { entry.create_instance(&create_info, None).unwrap() }
    }

    fn create_logical_device(
        entry: &Entry,
        instance: &Instance,
        surface: SurfaceKHR,
    ) -> (Device, QueueFamilyIndices, SwapChainSupportDetails) {
        let (physical_device, queue_indices, swap_chain_support_details) =
            Self::pick_physical_device(entry, instance, surface);

        let unique_indices = {
            let mut set = HashSet::new();
            set.insert(queue_indices.graphics_family);
            set.insert(queue_indices.present_family);
            set
        };

        let queue_create_infos: Vec<_> = unique_indices
            .into_iter()
            .map(|index| DeviceQueueCreateInfo {
                queue_count: 1,
                queue_family_index: index,
                p_queue_priorities: &1.0f32,

                ..Default::default()
            })
            .collect();

        let device_features: PhysicalDeviceFeatures = Default::default();

        let create_info = DeviceCreateInfo {
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,

            pp_enabled_extension_names: DEVICE_EXTENSIONS.map(CStr::as_ptr).as_ptr(),
            enabled_extension_count: DEVICE_EXTENSIONS.len() as u32,

            p_enabled_features: &device_features,

            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &create_info, None) }
            .expect("failed to create logical device!");
        (device, queue_indices, swap_chain_support_details)
    }

    fn find_queue_families(
        entry: &Entry,
        instance: &Instance,
        device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> PartialQueueFamilyIndices {
        let surface_fn = Surface::new(&entry, &instance);

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };

        let mut result: PartialQueueFamilyIndices = Default::default();

        for (index, queue) in props.into_iter().enumerate() {
            let index = index as u32;
            if queue.queue_flags.contains(QueueFlags::GRAPHICS) {
                result.graphics_family = Some(index);
            }
            if unsafe {
                surface_fn
                    .get_physical_device_surface_support(device, index, surface)
                    .unwrap_or(false)
            } {
                result.present_family = Some(index);
            }

            if result.is_total() {
                break;
            }
        }

        result
    }

    fn pick_physical_device(
        entry: &Entry,
        instance: &Instance,
        surface: SurfaceKHR,
    ) -> (PhysicalDevice, QueueFamilyIndices, SwapChainSupportDetails) {
        let is_device_suitable = |device: PhysicalDevice| {
            if !Self::check_device_extension_support(instance, device) {
                return None;
            }

            let swap_chain_support =
                Self::query_swap_chain_support(entry, instance, device, surface);
            if swap_chain_support.formats.is_empty() || swap_chain_support.present_modes.is_empty()
            {
                return None;
            }

            Some((
                device,
                Self::find_queue_families(entry, instance, device, surface).as_total()?,
                swap_chain_support,
            ))
        };

        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        if devices.is_empty() {
            panic!("failed to find GPUs with Vulkan support!");
        }

        devices
            .into_iter()
            .find_map(is_device_suitable)
            .expect("failed to find a suitable GPU!")
    }

    fn check_device_extension_support(instance: &Instance, device: PhysicalDevice) -> bool {
        let exts =
            unsafe { instance.enumerate_device_extension_properties(device) }.unwrap_or_default();
        let exts: Vec<_> = exts
            .iter()
            .map(|prop| unsafe { name_to_cstr(&prop.extension_name) })
            .collect();

        DEVICE_EXTENSIONS.into_iter().all(|ext| exts.contains(&ext))
    }

    fn query_swap_chain_support(
        entry: &Entry,
        instance: &Instance,
        device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> SwapChainSupportDetails {
        let surface_fn = Surface::new(entry, instance);
        let capabilities =
            unsafe { surface_fn.get_physical_device_surface_capabilities(device, surface) }
                .unwrap_or_default();
        let formats = unsafe { surface_fn.get_physical_device_surface_formats(device, surface) }
            .unwrap_or_default();
        let present_modes =
            unsafe { surface_fn.get_physical_device_surface_present_modes(device, surface) }
                .unwrap_or_default();

        SwapChainSupportDetails {
            capabilities,
            formats,
            present_modes,
        }
    }

    fn create_swap_chain(
        swapchain: &Swapchain,
        window: &Window,
        surface: SurfaceKHR,
        swap_chain_support: SwapChainSupportDetails,
        queue_family_indices: QueueFamilyIndices,
    ) -> SwapChainData {
        let surface_format = Self::choose_swap_surface_format(&swap_chain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swap_chain_support.present_modes);
        let extent = Self::choose_swap_extent(window, &swap_chain_support.capabilities);

        let image_count = (swap_chain_support.capabilities.min_image_count + 1).clamp(
            swap_chain_support.capabilities.min_image_count,
            if swap_chain_support.capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                swap_chain_support.capabilities.max_image_count
            },
        );

        let queue_indices = [
            queue_family_indices.graphics_family,
            queue_family_indices.present_family,
        ];

        let queue_indices: &[u32] =
            if queue_family_indices.graphics_family == queue_family_indices.present_family {
                &[]
            } else {
                &queue_indices
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

            pre_transform: swap_chain_support.capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: TRUE,
            old_swapchain: SwapchainKHR::null(),
            ..Default::default()
        };

        let swap_chain = unsafe { swapchain.create_swapchain(&create_info, None) }
            .expect("failed to create swapchain");

        let images = unsafe { swapchain.get_swapchain_images(swap_chain) }
            .expect("failed to get swapchain images");

        SwapChainData {
            swap_chain,
            format: surface_format.format,
            extent,
            images,
        }
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

    fn choose_swap_extent(window: &Window, capabilities: &SurfaceCapabilitiesKHR) -> Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        } else {
            let size = window.inner_size();

            Extent2D {
                width: size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
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
                    format: swap_chain_data.format,

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

    fn create_graphics_pipeline(
        device: &Device,
        swap_chain_data: &SwapChainData,
        render_pass: RenderPass,
    ) -> (Pipeline, PipelineLayout) {
        let vert_shader_module = Self::create_shader_module(device, include_shader!("vert.spv"));
        let vert_shader_stage_info = PipelineShaderStageCreateInfo {
            stage: ShaderStageFlags::VERTEX,
            module: vert_shader_module,
            p_name: cstr!("main").as_ptr(),

            ..Default::default()
        };

        let frag_shader_module = Self::create_shader_module(device, include_shader!("frag.spv"));
        let frag_shader_stage_info = PipelineShaderStageCreateInfo {
            stage: ShaderStageFlags::FRAGMENT,
            module: frag_shader_module,
            p_name: cstr!("main").as_ptr(),

            ..Default::default()
        };

        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let dynamic_states = [DynamicState::VIEWPORT, DynamicState::SCISSOR];
        let dynamic_state = PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),

            ..Default::default()
        };

        let vertex_input_info = PipelineVertexInputStateCreateInfo::default();
        let input_assembly = PipelineInputAssemblyStateCreateInfo {
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: FALSE,

            ..Default::default()
        };

        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: swap_chain_data.extent.width as f32,
            height: swap_chain_data.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: swap_chain_data.extent,
        };
        let viewport_state = PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: FALSE,
            rasterizer_discard_enable: FALSE,

            polygon_mode: PolygonMode::FILL,
            line_width: 1.0,

            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::CLOCKWISE,

            depth_bias_enable: FALSE,

            ..Default::default()
        };

        let multisampling = PipelineMultisampleStateCreateInfo {
            sample_shading_enable: FALSE,
            rasterization_samples: SampleCountFlags::TYPE_1,

            ..Default::default()
        };

        let color_blend_attachments = [PipelineColorBlendAttachmentState {
            color_write_mask: ColorComponentFlags::RGBA,
            blend_enable: FALSE,

            ..Default::default()
        }];

        let color_blending = PipelineColorBlendStateCreateInfo {
            logic_op_enable: FALSE,

            attachment_count: color_blend_attachments.len() as u32,
            p_attachments: color_blend_attachments.as_ptr(),

            ..Default::default()
        };

        let pipeline_layout_info = PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .expect("failed to create pipeline layout!");

        let pipeline_infos = [GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),

            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,

            layout: pipeline_layout,

            render_pass,
            subpass: 0,

            ..Default::default()
        }];

        let pipeline = unsafe {
            device.create_graphics_pipelines(PipelineCache::null(), &pipeline_infos, None)
        }
        .expect("failed to create graphics pipeline!")[0];

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        (pipeline, pipeline_layout)
    }

    fn create_shader_module(device: &Device, code: &[u8]) -> ShaderModule {
        let create_info = ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32,

            ..Default::default()
        };

        unsafe { device.create_shader_module(&create_info, None) }
            .expect("failed to create shader module")
    }

    fn create_render_pass(device: &Device, swap_chain_data: &SwapChainData) -> RenderPass {
        let color_attachments = [AttachmentDescription {
            format: swap_chain_data.format,
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

        let render_pass_create_info = RenderPassCreateInfo {
            attachment_count: color_attachments.len() as u32,
            p_attachments: color_attachments.as_ptr(),

            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),

            ..Default::default()
        };

        unsafe { device.create_render_pass(&render_pass_create_info, None) }
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
            .enumerate()
            .map(|(index, &img_view)| {
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
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|&fbuf| self.device.destroy_framebuffer(fbuf, None));

            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            self.image_views
                .iter()
                .for_each(|&img_view| self.device.destroy_image_view(img_view, None));

            Swapchain::new(&self.instance, &self.device)
                .destroy_swapchain(self.swap_chain_data.swap_chain, None);
            self.device.destroy_device(None);
            Surface::new(&self.entry, &self.instance).destroy_surface(self.surface, None);

            #[cfg(debug_assertions)]
            ManuallyDrop::drop(&mut self.messenger);

            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();

    let entry = Entry::linked();

    let _app = HelloTriangleApplication::new(entry, &window);

    return;

    let mat = glm::mat2(1., 2., 3., 4.);
    let v = glm::vec2(1., 2.);
    let test = mat * v;
    println!("{test:?}");

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    event_loop
        .run(|event, elwt| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                println!("Close clicked");
                elwt.exit();
            }
            Event::AboutToWait => elwt.exit(),
            _ => {
                println!("{event:?}")
            }
        })
        .unwrap();
}

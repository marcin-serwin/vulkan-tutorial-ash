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

const MAX_FRAMES_IN_FLIGHT: usize = 2;

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
    chain: SwapchainKHR,
    images: Vec<Image>,
    format: Format,
    extent: Extent2D,
}

struct SyncObjects {
    image_available_semaphores: [Semaphore; MAX_FRAMES_IN_FLIGHT],
    render_finished_semaphores: [Semaphore; MAX_FRAMES_IN_FLIGHT],
    in_flight_fences: [Fence; MAX_FRAMES_IN_FLIGHT],
}

struct DeviceInfo {
    device: Device,
    swap_chain_support: SwapChainSupportDetails,
    queue_indices: QueueFamilyIndices,
}

struct HelloTriangleApplication<'a> {
    entry: Entry,
    instance: Instance,
    window: &'a Window,
    surface: SurfaceKHR,
    device_info: DeviceInfo,
    swapchain_fns: Swapchain,
    occluded: bool,

    queue_family: QueueFamily,
    swap_chain: SwapChainData,
    image_views: Vec<ImageView>,

    render_pass: RenderPass,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,

    framebuffers: Vec<Framebuffer>,

    command_pool: CommandPool,
    command_buffers: [CommandBuffer; MAX_FRAMES_IN_FLIGHT],

    sync_objects: SyncObjects,
    current_frame: usize,

    #[cfg(debug_assertions)]
    messenger: ManuallyDrop<Messenger>,
}

impl<'a> HelloTriangleApplication<'a> {
    fn new(entry: Entry, window: &'a Window) -> Self {
        let occluded = {
            let winit::dpi::PhysicalSize { width, height } = window.inner_size();

            width == 0 || height == 0
        };
        let instance = Self::create_instance(&entry);
        #[cfg(debug_assertions)]
        let messenger = ManuallyDrop::new(unsafe { Messenger::new(&entry, &instance) });
        let surface = Self::create_surface(&entry, &instance, window);
        let device_info = Self::create_logical_device(&entry, &instance, surface);
        let device = &device_info.device;

        let queue_family = QueueFamily {
            graphics_queue: unsafe {
                device.get_device_queue(device_info.queue_indices.graphics_family, 0)
            },
            present_queue: unsafe {
                device.get_device_queue(device_info.queue_indices.present_family, 0)
            },
        };

        let swapchain_fns = Swapchain::new(&instance, &device);
        let swap_chain = Self::create_swap_chain(
            &swapchain_fns,
            window,
            surface,
            &device_info.swap_chain_support,
            &device_info.queue_indices,
        );
        let image_views = Self::create_image_views(&device, &swap_chain);

        let render_pass = Self::create_render_pass(&device, &swap_chain);
        let framebuffers =
            Self::create_framebuffers(&device, &swap_chain, &image_views, render_pass);

        let (pipeline, pipeline_layout) = Self::create_graphics_pipeline(&device, render_pass);

        let command_pool = Self::create_command_pool(&device, &device_info.queue_indices);
        let command_buffers = Self::create_command_buffers(&device, command_pool);

        let sync_objects = Self::create_sync_objects(&device);

        Self {
            entry,
            instance,
            window,
            surface,
            device_info,
            swapchain_fns,
            occluded,

            queue_family,
            swap_chain,
            image_views,

            render_pass,

            pipeline_layout,
            pipeline,

            framebuffers,

            command_pool,
            command_buffers,

            sync_objects,
            current_frame: 0,

            #[cfg(debug_assertions)]
            messenger,
        }
    }

    fn draw_frame(&mut self) {
        if self.occluded {
            return;
        }
        let in_flight = self.sync_objects.in_flight_fences[self.current_frame];
        let img_available = self.sync_objects.image_available_semaphores[self.current_frame];
        let render_finished = self.sync_objects.render_finished_semaphores[self.current_frame];
        let command_buffer = self.command_buffers[self.current_frame];

        unsafe {
            self.device_info
                .device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .expect("failed to wait for fence");
        }

        let image_index = unsafe {
            match self.swapchain_fns.acquire_next_image(
                self.swap_chain.chain,
                u64::MAX,
                img_available,
                Fence::null(),
            ) {
                Ok((index, _)) => index,
                Err(Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swap_chain();
                    return;
                }
                _ => panic!("failed to acquire next image from swap chain"),
            }
        };

        unsafe {
            self.device_info
                .device
                .reset_fences(&[in_flight])
                .expect("failed to reset fence");
        }

        unsafe {
            self.device_info
                .device
                .reset_command_buffer(command_buffer, CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");

            self.record_command_buffer(command_buffer, image_index);
        }

        let wait_semaphores = [img_available];
        let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_buffers = [command_buffer];
        let signal_semaphores = [render_finished];

        let submit_info = SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device_info
                .device
                .queue_submit(self.queue_family.graphics_queue, &[*submit_info], in_flight)
                .expect("failed to submit draw command buffer!")
        };

        let swapchains = [self.swap_chain.chain];
        let image_indices = [image_index];
        let present_info = PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            match self
                .swapchain_fns
                .queue_present(self.queue_family.present_queue, &present_info)
            {
                Ok(false) => (),
                Ok(true) | Err(Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swap_chain(),
                _ => panic!("failed to present queue"),
            }
        }
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
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
    ) -> DeviceInfo {
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

        DeviceInfo {
            device,
            queue_indices,
            swap_chain_support: swap_chain_support_details,
        }
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
        swap_chain_support: &SwapChainSupportDetails,
        queue_family_indices: &QueueFamilyIndices,
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
            chain: swap_chain,
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

        let dependencies = [SubpassDependency {
            src_subpass: SUBPASS_EXTERNAL,
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

    fn create_command_pool(
        device: &Device,
        queue_family_indices: &QueueFamilyIndices,
    ) -> CommandPool {
        let pool_info = CommandPoolCreateInfo {
            flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: queue_family_indices.graphics_family,

            ..Default::default()
        };

        unsafe { device.create_command_pool(&pool_info, None) }
            .expect("failed to create framebuffer!")
    }

    fn create_command_buffers(
        device: &Device,
        command_pool: CommandPool,
    ) -> [CommandBuffer; MAX_FRAMES_IN_FLIGHT] {
        let alloc_info = CommandBufferAllocateInfo {
            command_pool,
            level: CommandBufferLevel::PRIMARY,
            command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };

        unsafe { device.allocate_command_buffers(&alloc_info) }
            .expect("failed to allocate command buffers!")
            .try_into()
            .expect("incorrect number of buffers returned")
    }

    fn record_command_buffer(&self, command_buffer: CommandBuffer, image_index: u32) {
        let begin_info = CommandBufferBeginInfo::default();

        unsafe {
            self.device_info
                .device
                .begin_command_buffer(command_buffer, &begin_info)
        }
        .expect("failed to allocate command buffers!");

        let clear_colors = [ClearValue {
            color: ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = RenderPassBeginInfo {
            render_pass: self.render_pass,
            framebuffer: self.framebuffers[image_index as usize],
            render_area: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: self.swap_chain.extent,
            },

            clear_value_count: clear_colors.len() as u32,
            p_clear_values: clear_colors.as_ptr(),

            ..Default::default()
        };

        let viewports = [Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swap_chain.extent.width as f32,
            height: self.swap_chain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: self.swap_chain.extent,
        }];

        unsafe {
            self.device_info.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                SubpassContents::INLINE,
            );

            self.device_info.device.cmd_bind_pipeline(
                command_buffer,
                PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.device_info
                .device
                .cmd_set_viewport(command_buffer, 0, &viewports);
            self.device_info
                .device
                .cmd_set_scissor(command_buffer, 0, &scissors);
            self.device_info.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            self.device_info.device.cmd_end_render_pass(command_buffer);

            self.device_info
                .device
                .end_command_buffer(command_buffer)
                .expect("failed to record command buffer!");
        }
    }

    fn create_sync_objects(device: &Device) -> SyncObjects {
        let semaphore_info = SemaphoreCreateInfo::default();
        let fence_info = FenceCreateInfo {
            flags: FenceCreateFlags::SIGNALED,

            ..Default::default()
        };

        const UNIT_ARRAY: [(); MAX_FRAMES_IN_FLIGHT] = [(); MAX_FRAMES_IN_FLIGHT];
        let semaphore_init = |_| unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .expect("failed to create semaphore")
        };
        let fence_init = |_| unsafe {
            device
                .create_fence(&fence_info, None)
                .expect("failed to create fence")
        };

        SyncObjects {
            image_available_semaphores: UNIT_ARRAY.map(semaphore_init),
            render_finished_semaphores: UNIT_ARRAY.map(semaphore_init),
            in_flight_fences: UNIT_ARRAY.map(fence_init),
        }
    }

    unsafe fn cleanup_swap_chain(&mut self) {
        self.framebuffers
            .iter()
            .for_each(|&fbuf| self.device_info.device.destroy_framebuffer(fbuf, None));

        self.device_info
            .device
            .destroy_render_pass(self.render_pass, None);

        self.image_views
            .iter()
            .for_each(|&img_view| self.device_info.device.destroy_image_view(img_view, None));

        self.swapchain_fns
            .destroy_swapchain(self.swap_chain.chain, None);
    }

    fn recreate_swap_chain(&mut self) {
        if self.occluded {
            return;
        }
        unsafe {
            self.device_info
                .device
                .device_wait_idle()
                .expect("failed while waiting for device idle")
        };

        unsafe { self.cleanup_swap_chain() };

        self.swap_chain = Self::create_swap_chain(
            &self.swapchain_fns,
            self.window,
            self.surface,
            &self.device_info.swap_chain_support,
            &self.device_info.queue_indices,
        );
        self.image_views = Self::create_image_views(&self.device_info.device, &self.swap_chain);
        self.render_pass = Self::create_render_pass(&self.device_info.device, &self.swap_chain);
        self.framebuffers = Self::create_framebuffers(
            &self.device_info.device,
            &self.swap_chain,
            &self.image_views,
            self.render_pass,
        );
    }
}

impl<'a> Drop for HelloTriangleApplication<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device_info
                .device
                .device_wait_idle()
                .expect("failed while waiting for device idle");

            self.sync_objects
                .image_available_semaphores
                .iter()
                .chain(self.sync_objects.render_finished_semaphores.iter())
                .for_each(|&sem| self.device_info.device.destroy_semaphore(sem, None));

            self.sync_objects
                .in_flight_fences
                .iter()
                .for_each(|&fence| self.device_info.device.destroy_fence(fence, None));

            self.cleanup_swap_chain();

            self.device_info
                .device
                .destroy_command_pool(self.command_pool, None);

            self.device_info
                .device
                .destroy_pipeline(self.pipeline, None);
            self.device_info
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device_info.device.destroy_device(None);
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

    let mut app = HelloTriangleApplication::new(entry, &window);

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
            Event::WindowEvent {
                window_id: _,
                event: WindowEvent::Resized(size),
            } => {
                println!("Resized {size:?}");
                app.occluded = size.width == 0 || size.height == 0;
                app.recreate_swap_chain();
            }
            Event::WindowEvent {
                window_id: _,
                event: WindowEvent::Occluded(is_occluded),
            } => {
                println!("Occluded: {is_occluded:?}");
                app.occluded = is_occluded;
            }
            Event::WindowEvent {
                window_id: _,
                event: WindowEvent::RedrawRequested,
            } => {
                println!("Redraw requested");
                app.draw_frame();
            }
            _ => {
                // println!("{event:?}");
            }
        })
        .unwrap();
}

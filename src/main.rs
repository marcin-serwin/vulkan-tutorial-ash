use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
#[cfg_attr(not(debug_assertions), allow(unused_imports))]
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

unsafe fn name_to_cstr(name: &[c_char]) -> &CStr {
    return CStr::from_ptr(name.as_ptr());
}
const DEVICE_EXTENSIONS: [&CStr; 1] = [KhrSwapchainFn::name()];

#[cfg(not(debug_assertions))]
const LAYERS: [&CStr; 0] = [];
#[cfg(debug_assertions)]
const LAYERS: [&CStr; 1] =
    [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
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
    swap_chain_images: Vec<Image>,
    swap_chain_format: Format,
    swap_chain_extent: Extent2D,
}

struct HelloTriangleApplication {
    entry: Entry,
    instance: Instance,
    surface: SurfaceKHR,
    device: Device,
    queue_family: QueueFamily,
    swap_chain_data: SwapChainData,

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

        let swapchain = Swapchain::new(&instance, &device);
        let (swap_chain, swap_chain_format, swap_chain_extent) = Self::create_swap_chain(
            &swapchain,
            window,
            surface,
            swap_chain_support_details,
            queue_indices,
        );
        let swap_chain_images = unsafe { swapchain.get_swapchain_images(swap_chain) }
            .expect("failed to get swapchain images");

        Self {
            entry,
            instance,
            surface,
            device,
            queue_family,
            swap_chain_data: SwapChainData {
                swap_chain,
                swap_chain_format,
                swap_chain_extent,
                swap_chain_images,
            },

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
            s_type: StructureType::WAYLAND_SURFACE_CREATE_INFO_KHR,
            display: display_handle.display.as_ptr(),
            surface: window_handle.surface.as_ptr(),
            ..Default::default()
        };

        let surface = unsafe { surface.create_wayland_surface(&create_info, None) }
            .expect("failed to create wayland surface");
        surface
    }

    fn create_instance(entry: &Entry) -> Instance {
        const APP_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"MyTest\0") };

        let app_info = ApplicationInfo {
            api_version: make_api_version(0, 1, 0, 0),
            p_application_name: APP_NAME.as_ptr(),

            s_type: StructureType::APPLICATION_INFO,
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

                s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
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

            s_type: StructureType::DEVICE_CREATE_INFO,
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
    ) -> (SwapchainKHR, Format, Extent2D) {
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
            s_type: StructureType::SWAPCHAIN_CREATE_INFO_KHR,
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

        let swapchain = unsafe { swapchain.create_swapchain(&create_info, None) }
            .expect("failed to create swapchain");

        (swapchain, surface_format.format, extent)
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
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            ManuallyDrop::drop(&mut self.messenger);

            Swapchain::new(&self.instance, &self.device)
                .destroy_swapchain(self.swap_chain_data.swap_chain, None);
            self.device.destroy_device(None);
            Surface::new(&self.entry, &self.instance).destroy_surface(self.surface, None);
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

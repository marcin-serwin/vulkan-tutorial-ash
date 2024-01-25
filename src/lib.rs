mod app_state;
mod surface;
mod utils;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::{c_void, CStr};
use std::mem::ManuallyDrop;
use surface::{SwapChain, SwapChainData, WindowSurface};

use ash::extensions::ext::DebugUtils;
use ash::{extensions::khr::Surface, vk::*, Device, Entry, Instance};
use nalgebra::*;

use winit::dpi::PhysicalSize;
use winit::window::Window;

use app_state::AppState;

use image::io::Reader as ImageReader;
use image::Rgba;

use utils::{cstr, include_bytes_as_array, name_to_cstr, offset_of};

macro_rules! include_shader {
    ($name:literal) => {
        include_bytes_as_array!(u32, concat!(env!("OUT_DIR"), "/", $name))
    };
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

const DEVICE_EXTENSIONS: [&CStr; 1 + cfg!(target_os = "macos") as usize] = [
    KhrSwapchainFn::name(),
    #[cfg(target_os = "macos")]
    KhrPortabilitySubsetFn::name(),
];

const LAYERS: &[&CStr] = &[
    #[cfg(debug_assertions)]
    cstr!("VK_LAYER_KHRONOS_validation"),
];
fn get_validation_layers(entry: &Entry) -> [&'static CStr; LAYERS.len()] {
    let layers: [&CStr; LAYERS.len()] = LAYERS
        .try_into()
        .expect("This is a constant slice with known length");
    if layers.is_empty() {
        return layers;
    }
    let supported_layers = entry.enumerate_instance_layer_properties().unwrap();

    let names: Vec<&CStr> = supported_layers
        .iter()
        .map(|layer| name_to_cstr(&layer.layer_name))
        .collect();

    assert!(
        !layers.iter().any(|ext| !names.contains(ext)),
        "Required validation layer not found"
    );

    layers
}

const EXTENSIONS: &[&CStr] = &[
    KhrSurfaceFn::name(),
    #[cfg(target_os = "linux")]
    KhrWaylandSurfaceFn::name(),
    #[cfg(target_os = "macos")]
    ExtMetalSurfaceFn::name(),
    #[cfg(target_os = "macos")]
    KhrPortabilityEnumerationFn::name(),
    #[cfg(target_os = "macos")]
    KhrGetPhysicalDeviceProperties2Fn::name(),
    #[cfg(debug_assertions)]
    ExtDebugUtilsFn::name(),
];

fn get_extensions(entry: &Entry) -> [&'static CStr; EXTENSIONS.len()] {
    let extensions: [&CStr; EXTENSIONS.len()] = EXTENSIONS
        .try_into()
        .expect("This is a constant slice with known length");
    if extensions.is_empty() {
        return extensions;
    }
    let supported_extensions = entry.enumerate_instance_extension_properties(None).unwrap();
    let names: Vec<&CStr> = supported_extensions
        .iter()
        .map(|ext| name_to_cstr(&ext.extension_name))
        .collect();

    assert!(
        !extensions.iter().any(|ext| !names.contains(ext)),
        "Required extension not found"
    );

    extensions
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
        assert!(
            !message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR),
            "Validation error"
        );
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
struct PartialQueueFamilyIndices {
    graphics: Option<u32>,
    present: Option<u32>,
    transfer: Option<u32>,
}

impl PartialQueueFamilyIndices {
    fn is_total(&self) -> bool {
        self.as_total().is_some()
    }

    fn as_total(&self) -> Option<QueueFamilyIndices> {
        Some(QueueFamilyIndices {
            graphics: self.graphics?,
            present: self.present?,
            transfer: self.transfer?,
        })
    }
}

#[derive(Debug, Default)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
    transfer: u32,
}

pub struct QueueWrapper {
    pub handle: Queue,
    index: u32,
}
impl QueueWrapper {
    fn new(device: &Device, index: u32) -> Self {
        Self {
            handle: unsafe { device.get_device_queue(index, 0) },
            index,
        }
    }
}

struct SyncObjects {
    image_available_semaphore: Semaphore,
    render_finished_semaphore: Semaphore,
    in_flight_fence: Fence,
}

#[derive(Debug, Default)]
struct SwapChainSupportDetails {
    capabilities: SurfaceCapabilitiesKHR,
    formats: Vec<SurfaceFormatKHR>,
    present_modes: Vec<PresentModeKHR>,
}

struct DeviceInfo {
    device: Device,
    physical_device: PhysicalDevice,
    swap_chain_support: SwapChainSupportDetails,
    queue_indices: QueueFamilyIndices,
    physical_memory_properties: PhysicalDeviceMemoryProperties,
}

impl DeviceInfo {
    fn query_swap_chain_support(
        surface_fn: &Surface,
        device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> SwapChainSupportDetails {
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
}

struct TransferCommandPool {
    command_pool: CommandPool,
    queue: QueueWrapper,
}

impl TransferCommandPool {
    fn new(device: &Device, queue: QueueWrapper) -> Self {
        let pool_info = CommandPoolCreateInfo::builder()
            .queue_family_index(queue.index)
            .flags(CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .expect("failed to create command pool");

        Self {
            command_pool,
            queue,
        }
    }

    fn copy_buffer(
        &self,
        device: &Device,
        src_buffer: Buffer,
        dst_buffer: Buffer,
        size: DeviceSize,
    ) {
        let cmd_buffer_info = CommandBufferAllocateInfo::builder()
            .level(CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let cmd_buffer = unsafe { device.allocate_command_buffers(&cmd_buffer_info) }.unwrap()[0];

        let cmd_buffer_begin_info =
            CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(cmd_buffer, &cmd_buffer_begin_info) }.unwrap();

        let copy_region = BufferCopy::builder().size(size);
        unsafe { device.cmd_copy_buffer(cmd_buffer, src_buffer, dst_buffer, &[*copy_region]) };

        unsafe { device.end_command_buffer(cmd_buffer) }.unwrap();

        let cmd_buffers = [cmd_buffer];
        let submit_info = SubmitInfo::builder().command_buffers(&cmd_buffers);

        unsafe { device.queue_submit(self.queue.handle, &[*submit_info], Fence::null()) }.unwrap();
        unsafe { device.queue_wait_idle(self.queue.handle) }.unwrap();

        unsafe { device.free_command_buffers(self.command_pool, &cmd_buffers) };
    }

    unsafe fn cleanup(&mut self, device: &Device) {
        device.destroy_command_pool(self.command_pool, None);
    }
}

struct BufferWrapper {
    buffer: Buffer,
    memory: DeviceMemory,
}

impl BufferWrapper {
    fn new(
        device: &Device,
        mem_props: &PhysicalDeviceMemoryProperties,
        size: DeviceSize,
        mem_prop_flags: MemoryPropertyFlags,
        usage_flags: BufferUsageFlags,
    ) -> Self {
        let buffer_info = BufferCreateInfo::builder()
            .size(size)
            .usage(usage_flags)
            .sharing_mode(SharingMode::EXCLUSIVE);

        let buffer =
            unsafe { device.create_buffer(&buffer_info, None) }.expect("failed to create buffer");

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let find_memory_buffer = |type_filter: u32, props: MemoryPropertyFlags| -> u32 {
            mem_props
                .memory_types
                .iter()
                .take(mem_props.memory_type_count as usize)
                .enumerate()
                .find(|(index, typ)| {
                    (type_filter & (1 << index)) != 0 && typ.property_flags.contains(props)
                })
                .expect("failed to find suitable memory type")
                .0 as u32
        };

        let alloc_info = MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(find_memory_buffer(
                mem_requirements.memory_type_bits,
                mem_prop_flags,
            ));

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .expect("failed to allocate vertex buffer memory");

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .expect("failed to bind buffer memory");

        Self { buffer, memory }
    }

    fn create_buffer_with_staging<T>(
        device: &DeviceInfo,
        data: &[T],
        transfer_cmd_pool: &TransferCommandPool,
        usage: BufferUsageFlags,
    ) -> BufferWrapper {
        let size = std::mem::size_of_val(data) as DeviceSize;

        let BufferWrapper {
            buffer: staging_buffer,
            memory: staging_memory,
        } = BufferWrapper::new(
            &device.device,
            &device.physical_memory_properties,
            size,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            BufferUsageFlags::TRANSFER_SRC,
        );

        let map_ptr = unsafe {
            device
                .device
                .map_memory(staging_memory, 0, size, MemoryMapFlags::empty())
        }
        .expect("failed to map memory");

        unsafe {
            data.as_ptr()
                .copy_to_nonoverlapping(map_ptr.cast(), data.len());
        };

        unsafe { device.device.unmap_memory(staging_memory) };

        let result @ BufferWrapper { buffer, .. } = BufferWrapper::new(
            &device.device,
            &device.physical_memory_properties,
            size,
            MemoryPropertyFlags::DEVICE_LOCAL,
            BufferUsageFlags::TRANSFER_DST | usage,
        );

        transfer_cmd_pool.copy_buffer(&device.device, staging_buffer, buffer, size);

        unsafe {
            device.device.destroy_buffer(staging_buffer, None);
            device.device.free_memory(staging_memory, None);
        }

        result
    }

    fn create_mapped_buffer(device: &DeviceInfo, size: DeviceSize) -> (Self, *mut c_void) {
        let buffer = Self::new(
            &device.device,
            &device.physical_memory_properties,
            size,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            BufferUsageFlags::UNIFORM_BUFFER,
        );

        let ptr = unsafe {
            device
                .device
                .map_memory(buffer.memory, 0, size, MemoryMapFlags::empty())
                .unwrap()
        };

        (buffer, ptr)
    }

    unsafe fn cleanup(&mut self, device: &Device) {
        device.free_memory(self.memory, None);
        device.destroy_buffer(self.buffer, None);
    }
}

struct InFlightBuffers {
    sync_objects: SyncObjects,
    command_buffer: CommandBuffer,
    uniform_buffer: BufferWrapper,
    ub_map: *mut UniformBufferObject,
    descriptor_set: DescriptorSet,
}

impl InFlightBuffers {
    fn new(
        device: &DeviceInfo,
        command_buffer: CommandBuffer,
        descriptor_set: DescriptorSet,
        (uniform_buffer, ub_map): (BufferWrapper, *mut c_void),
    ) -> Self {
        let sync_objects = Self::create_sync_objects(&device.device);

        Self {
            command_buffer,
            sync_objects,
            uniform_buffer,
            descriptor_set,
            ub_map: ub_map.cast(),
        }
    }

    fn create_sync_objects(device: &Device) -> SyncObjects {
        let semaphore_info = SemaphoreCreateInfo::default();
        let fence_info = FenceCreateInfo {
            flags: FenceCreateFlags::SIGNALED,

            ..Default::default()
        };

        let semaphore_init = || unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .expect("failed to create semaphore")
        };
        let fence_init = || unsafe {
            device
                .create_fence(&fence_info, None)
                .expect("failed to create fence")
        };

        SyncObjects {
            image_available_semaphore: semaphore_init(),
            render_finished_semaphore: semaphore_init(),
            in_flight_fence: fence_init(),
        }
    }

    unsafe fn cleanup(&mut self, device: &Device) {
        let SyncObjects {
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        } = self.sync_objects;

        device.destroy_semaphore(image_available_semaphore, None);
        device.destroy_semaphore(render_finished_semaphore, None);
        device.destroy_fence(in_flight_fence, None);

        self.uniform_buffer.cleanup(device);
    }
}

struct GraphicsCommandPool {
    descriptor_pool: DescriptorPool,
    command_pool: CommandPool,

    in_flight_buffers: [InFlightBuffers; MAX_FRAMES_IN_FLIGHT],
    queue: QueueWrapper,

    current_frame: RefCell<usize>,
}

impl GraphicsCommandPool {
    fn new(
        device: &DeviceInfo,
        queue: QueueWrapper,
        descriptor_set_layout: DescriptorSetLayout,
    ) -> Self {
        let cmd_pool_info = CommandPoolCreateInfo {
            flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: queue.index,

            ..Default::default()
        };
        let command_pool = unsafe { device.device.create_command_pool(&cmd_pool_info, None) }
            .expect("failed to create command pool");

        let command_buffers = Self::create_command_buffers(&device.device, command_pool);
        let uniform_buffers: Vec<_> = command_buffers
            .iter()
            .map(|_| {
                BufferWrapper::create_mapped_buffer(
                    device,
                    std::mem::size_of::<UniformBufferObject>() as DeviceSize,
                )
            })
            .collect();

        let descriptor_pool = Self::create_descriptor_pool(&device.device);
        let descriptor_sets = Self::create_descriptor_sets(
            &device.device,
            descriptor_pool,
            descriptor_set_layout,
            uniform_buffers.iter().map(|buf| &buf.0),
        );
        let in_flight_buffers = unsafe {
            command_buffers
                .into_iter()
                .zip(uniform_buffers)
                .zip(descriptor_sets)
                .map(|((cmd_buffer, uniform_buffer), descriptor_set)| {
                    InFlightBuffers::new(device, cmd_buffer, descriptor_set, uniform_buffer)
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_unchecked()
        };

        Self {
            descriptor_pool,
            command_pool,
            queue,
            current_frame: RefCell::new(0),
            in_flight_buffers,
        }
    }

    fn create_descriptor_pool(device: &Device) -> DescriptorPool {
        let pool_sizes = [DescriptorPoolSize {
            ty: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
        }];
        let create_info = DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

        unsafe { device.create_descriptor_pool(&create_info, None) }.unwrap()
    }

    fn create_descriptor_sets<'a>(
        device: &Device,
        descriptor_pool: DescriptorPool,
        set_layout: DescriptorSetLayout,
        uniform_buffers: impl Iterator<Item = &'a BufferWrapper>,
    ) -> [DescriptorSet; MAX_FRAMES_IN_FLIGHT] {
        let layouts = [set_layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap();

        let buffer_infos: Vec<_> = uniform_buffers
            .map(|buf| {
                [*DescriptorBufferInfo::builder()
                    .buffer(buf.buffer)
                    .offset(0)
                    .range(std::mem::size_of::<UniformBufferObject>() as DeviceSize)]
            })
            .collect();

        let descriptor_writes: Vec<_> = descriptor_sets
            .iter()
            .zip(buffer_infos.iter())
            .map(|(&dst_set, buffer_info)| {
                *WriteDescriptorSet::builder()
                    .dst_set(dst_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(buffer_info)
            })
            .collect();

        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) }

        descriptor_sets.try_into().unwrap()
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

    fn draw_frame(
        &self,
        device: &Device,
        acquire_next_image: impl FnOnce(Semaphore) -> std::result::Result<u32, Result>,
        record_command_buffer: impl FnOnce(CommandBuffer, u32, &[DescriptorSet]),
        update_uniform_buffer: impl FnOnce(*mut UniformBufferObject),
    ) -> std::result::Result<(u32, [Semaphore; 1]), Result> {
        let mut current_frame = self.current_frame.borrow_mut();
        let InFlightBuffers {
            sync_objects:
                SyncObjects {
                    image_available_semaphore: img_available,
                    render_finished_semaphore: render_finished,
                    in_flight_fence: in_flight,
                },
            command_buffer,
            ub_map,
            descriptor_set,
            ..
        } = self.in_flight_buffers[*current_frame];

        unsafe {
            device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .expect("failed to wait for fence");
        }

        let image_index = acquire_next_image(img_available)?;

        unsafe {
            device
                .reset_fences(&[in_flight])
                .expect("failed to reset fence");
        }

        unsafe {
            device
                .reset_command_buffer(command_buffer, CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");
        }

        record_command_buffer(command_buffer, image_index, &[descriptor_set]);
        update_uniform_buffer(ub_map);

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
            device
                .queue_submit(self.queue.handle, &[*submit_info], in_flight)
                .expect("failed to submit draw command buffer!");
        }

        *current_frame = (*current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok((image_index, signal_semaphores))
    }

    unsafe fn cleanup(&mut self, device: &Device) {
        self.in_flight_buffers
            .iter_mut()
            .for_each(|buf| buf.cleanup(device));
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_command_pool(self.command_pool, None);
    }
}

pub struct HelloTriangleApplication {
    instance: Instance,
    surface_fn: Surface,
    surface: WindowSurface,
    device_info: DeviceInfo,
    pub occluded: bool,

    pub swap_chain: SwapChain,

    descriptor_set_layout: DescriptorSetLayout,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,

    vertex_buffer: BufferWrapper,
    index_buffer: BufferWrapper,

    graphics_command_pool: GraphicsCommandPool,
    present_queue: QueueWrapper,

    app_state: AppState,

    #[cfg(debug_assertions)]
    messenger: ManuallyDrop<Messenger>,
}

impl HelloTriangleApplication {
    #[must_use]
    pub fn new(entry: &Entry, window: &Window) -> Self {
        let occluded = {
            let winit::dpi::PhysicalSize { width, height } = window.inner_size();

            width == 0 || height == 0
        };
        let instance = Self::create_instance(entry);
        let surface_fn = Surface::new(entry, &instance);
        #[cfg(debug_assertions)]
        let messenger = ManuallyDrop::new(unsafe { Messenger::new(entry, &instance) });
        let surface = WindowSurface::new(entry, &instance, window);
        let device_info = Self::create_logical_device(&surface_fn, &instance, &surface);
        let device = &device_info.device;

        let mut transfer_command_pool = TransferCommandPool::new(
            device,
            QueueWrapper::new(device, device_info.queue_indices.transfer),
        );
        Self::create_texture_image();

        let vertex_buffer = BufferWrapper::create_buffer_with_staging(
            &device_info,
            VERTICES,
            &transfer_command_pool,
            BufferUsageFlags::VERTEX_BUFFER,
        );
        let index_buffer = BufferWrapper::create_buffer_with_staging(
            &device_info,
            INDICES,
            &transfer_command_pool,
            BufferUsageFlags::INDEX_BUFFER,
        );

        unsafe { transfer_command_pool.cleanup(device) };

        let swap_chain = surface::SwapChain::new(
            &instance,
            device,
            window,
            surface.handle,
            &device_info.swap_chain_support,
            &device_info.queue_indices,
        );

        let queue_indices = &device_info.queue_indices;

        let descriptor_set_layout = Self::create_descriptor_layout(device);
        let (pipeline, pipeline_layout) =
            Self::create_graphics_pipeline(device, swap_chain.render_pass, descriptor_set_layout);

        let graphics_command_pool = GraphicsCommandPool::new(
            &device_info,
            QueueWrapper::new(device, device_info.queue_indices.graphics),
            descriptor_set_layout,
        );
        let present_queue = QueueWrapper::new(device, queue_indices.present);
        let app_state = AppState::new(Isometry3::look_at_rh(
            &Point3::new(2.0, 2.0, 2.0),
            &Point3::origin(),
            &Vector3::z_axis(),
        ));

        Self {
            instance,
            surface_fn,
            surface,
            device_info,
            occluded,
            swap_chain,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            graphics_command_pool,
            present_queue,
            app_state,
            #[cfg(debug_assertions)]
            messenger,
        }
    }

    fn create_texture_image() {
        let img = ImageReader::with_format(
            std::io::Cursor::new(include_bytes!("textures/texture.jpg")),
            image::ImageFormat::Jpeg,
        )
        .decode()
        .unwrap()
        .into_rgba8();
        let image_size = img.as_raw().len() as DeviceSize;

        debug_assert_eq!(
            image_size,
            img.width() as DeviceSize
                * img.height() as DeviceSize
                * std::mem::size_of::<Rgba<u8>>() as DeviceSize
        );
    }

    pub fn window_resized(&mut self, size: PhysicalSize<u32>) {
        self.device_info.swap_chain_support = DeviceInfo::query_swap_chain_support(
            &self.surface_fn,
            self.device_info.physical_device,
            self.surface.handle,
        );
        self.swap_chain.recreate_swap_chain(
            &self.device_info.device,
            self.surface.handle,
            Some(size),
        );
    }

    pub fn draw_frame(&mut self) {
        if self.occluded {
            return;
        }

        self.app_state.update_app_state();

        let acquire_image_index = |img_available| self.swap_chain.acquire_next_image(img_available);
        let drawing_result = self.graphics_command_pool.draw_frame(
            &self.device_info.device,
            acquire_image_index,
            |buffer, image_index, dst_sets| {
                self.record_command_buffer(buffer, image_index, dst_sets);
            },
            |ub_map| self.update_uniform_buffer(ub_map, &self.swap_chain.data),
        );
        let (image_index, signal_semaphores) = match drawing_result {
            Ok(i) => i,
            Err(Result::ERROR_OUT_OF_DATE_KHR) => {
                self.swap_chain.recreate_swap_chain(
                    &self.device_info.device,
                    self.surface.handle,
                    None,
                );
                return;
            }
            _ => panic!("failed to draw frame"),
        };

        let swapchains = [self.swap_chain.data.chain];
        let image_indices = [image_index];
        let present_info = PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match self.swap_chain.present(&self.present_queue, &present_info) {
            Ok(false) => (),
            Ok(true) | Err(Result::ERROR_OUT_OF_DATE_KHR) => self.swap_chain.recreate_swap_chain(
                &self.device_info.device,
                self.surface.handle,
                None,
            ),
            _ => panic!("failed to present queue"),
        }
    }

    fn update_uniform_buffer(
        &self,
        buffer_map: *mut UniformBufferObject,
        swap_chain: &SwapChainData,
    ) {
        let model = Rotation3::from_axis_angle(&Vector3::z_axis(), self.app_state.model_rotation);

        let view = self.app_state.get_view();

        #[allow(clippy::cast_precision_loss)]
        let aspect_ratio = {
            let extent = swap_chain.extent;
            extent.width as f32 / extent.height as f32
        };

        let proj = Perspective3::new(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 10.0)
            .to_homogeneous();

        let ubo = UniformBufferObject::new(nalgebra::convert(model), view, proj);

        unsafe {
            std::ptr::addr_of!(ubo).copy_to_nonoverlapping(buffer_map, 1);
        }
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_info = ApplicationInfo {
            api_version: make_api_version(0, 1, 0, 0),
            p_application_name: cstr!("MyTest").as_ptr(),

            ..Default::default()
        };

        let extensions = get_extensions(entry).map(CStr::as_ptr);
        let validation_layers = get_validation_layers(entry).map(CStr::as_ptr);

        let create_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&validation_layers);

        #[cfg(target_os = "macos")]
        let create_info = create_info.flags(InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        #[cfg(debug_assertions)]
        let mut debug_create_info = DebugUtilsMessengerCreateInfoEXT {
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
        #[cfg(debug_assertions)]
        let create_info = create_info.push_next(&mut debug_create_info);

        // let exts = entry.enumerate_instance_extension_properties(None).unwrap();

        unsafe { entry.create_instance(&create_info, None).unwrap() }
    }

    fn create_logical_device(
        surface_fn: &Surface,
        instance: &Instance,
        surface: &WindowSurface,
    ) -> DeviceInfo {
        let (physical_device, queue_indices, swap_chain_support_details) =
            Self::pick_physical_device(surface_fn, instance, surface.handle);

        let unique_indices = {
            let mut set = HashSet::new();
            set.insert(queue_indices.graphics);
            set.insert(queue_indices.present);
            set.insert(queue_indices.transfer);
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

        let extensions = DEVICE_EXTENSIONS.map(CStr::as_ptr);
        let device_features = PhysicalDeviceFeatures::default();

        let create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&device_features);

        let device = unsafe { instance.create_device(physical_device, &create_info, None) }
            .expect("failed to create logical device!");

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        DeviceInfo {
            device,
            physical_device,
            queue_indices,
            swap_chain_support: swap_chain_support_details,
            physical_memory_properties: mem_props,
        }
    }

    fn find_queue_families(
        surface_fn: &Surface,
        instance: &Instance,
        device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> PartialQueueFamilyIndices {
        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };

        let mut result = PartialQueueFamilyIndices::default();

        for (index, queue) in props.into_iter().enumerate() {
            let index = index as u32;
            if queue.queue_flags.contains(QueueFlags::TRANSFER)
                && (cfg!(target_os = "macos") || !queue.queue_flags.contains(QueueFlags::GRAPHICS))
            {
                result.transfer = Some(index);
            }
            if queue.queue_flags.contains(QueueFlags::GRAPHICS) {
                result.graphics = Some(index);
            }
            if unsafe {
                surface_fn
                    .get_physical_device_surface_support(device, index, surface)
                    .unwrap_or(false)
            } {
                result.present = Some(index);
            }

            if result.is_total() {
                break;
            }
        }

        result
    }

    fn pick_physical_device(
        surface_fn: &Surface,
        instance: &Instance,
        surface: SurfaceKHR,
    ) -> (PhysicalDevice, QueueFamilyIndices, SwapChainSupportDetails) {
        let is_device_suitable = |device: PhysicalDevice| {
            if !Self::check_device_extension_support(instance, device) {
                return None;
            }

            let swap_chain_support =
                DeviceInfo::query_swap_chain_support(surface_fn, device, surface);
            if swap_chain_support.formats.is_empty() || swap_chain_support.present_modes.is_empty()
            {
                return None;
            }

            Some((
                device,
                Self::find_queue_families(surface_fn, instance, device, surface).as_total()?,
                swap_chain_support,
            ))
        };

        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };

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
            .map(|prop| name_to_cstr(&prop.extension_name))
            .collect();

        DEVICE_EXTENSIONS.into_iter().all(|ext| exts.contains(&ext))
    }

    fn create_graphics_pipeline(
        device: &Device,
        render_pass: RenderPass,
        descriptor_set_layout: DescriptorSetLayout,
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

        let binding_descriptions = Vertex::get_binding_descriptions();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let vertex_input_info = PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

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
            front_face: FrontFace::COUNTER_CLOCKWISE,

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

        let descriptor_set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .expect("failed to create pipeline layout!");

        let pipeline_infos = [GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),

            p_vertex_input_state: &*vertex_input_info,
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

    fn create_shader_module(device: &Device, code: &[u32]) -> ShaderModule {
        let create_info = ShaderModuleCreateInfo::builder().code(code);

        unsafe { device.create_shader_module(&create_info, None) }
            .expect("failed to create shader module")
    }

    fn record_command_buffer(
        &self,
        command_buffer: CommandBuffer,
        image_index: u32,
        descriptor_sets: &[DescriptorSet],
    ) {
        let device = &self.device_info.device;
        let begin_info = CommandBufferBeginInfo::default();

        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
            .expect("failed to allocate command buffers!");

        let clear_colors = [ClearValue {
            color: ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = RenderPassBeginInfo {
            render_pass: self.swap_chain.render_pass,
            framebuffer: self.swap_chain.framebuffers[image_index as usize],
            render_area: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: self.swap_chain.data.extent,
            },

            clear_value_count: clear_colors.len() as u32,
            p_clear_values: clear_colors.as_ptr(),

            ..Default::default()
        };

        #[allow(clippy::cast_precision_loss)]
        let viewports = [Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swap_chain.data.extent.width as f32,
            height: self.swap_chain.data.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: self.swap_chain.data.extent,
        }];

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(command_buffer, PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);

            let buffers = [self.vertex_buffer.buffer];
            device.cmd_bind_vertex_buffers(command_buffer, 0, &buffers, &[0]);

            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.buffer,
                0,
                IndexType::UINT16,
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                descriptor_sets,
                &[],
            );

            device.cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);
            device.cmd_end_render_pass(command_buffer);

            device
                .end_command_buffer(command_buffer)
                .expect("failed to record command buffer!");
        }
    }

    fn cleanup(&mut self) -> std::result::Result<(), Result> {
        unsafe {
            self.device_info.device.device_wait_idle()?;

            self.swap_chain.cleanup_swap_chain(&self.device_info.device);

            self.device_info
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.index_buffer.cleanup(&self.device_info.device);
            self.vertex_buffer.cleanup(&self.device_info.device);
            self.graphics_command_pool.cleanup(&self.device_info.device);

            self.device_info
                .device
                .destroy_pipeline(self.pipeline, None);
            self.device_info
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.surface.cleanup_surface();
            self.device_info.device.destroy_device(None);

            #[cfg(debug_assertions)]
            ManuallyDrop::drop(&mut self.messenger);

            self.instance.destroy_instance(None);
        }
        Ok(())
    }

    #[allow(clippy::result_large_err)]
    pub fn safe_drop(mut self) -> std::result::Result<(), (ManuallyDrop<Self>, Result)> {
        match self.cleanup() {
            Ok(res) => {
                std::mem::forget(self);
                Ok(res)
            }
            Err(err) => Err((ManuallyDrop::new(self), err)),
        }
    }

    fn create_descriptor_layout(device: &Device) -> DescriptorSetLayout {
        let ubo_layout_binding = DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX);

        let bindings = [*ubo_layout_binding];
        let layout_info = DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    pub fn handle_key_event(&mut self, event: &winit::event::KeyEvent) {
        self.app_state.handle_key_event(event);
    }

    pub fn handle_mouse_movement(&mut self, delta: (f64, f64)) {
        self.app_state.handle_mouse_movement(delta);
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        if self.cleanup().is_err() {
            if std::thread::panicking() {
                eprintln!("failed to cleanup app during panicking");
            } else {
                panic!("failed to cleanup app");
            }
        }
    }
}

#[derive(Debug)]
#[repr(align(16))]
struct M4(Matrix4<f32>);

#[derive(Debug)]
#[repr(C)]
struct UniformBufferObject {
    _foo: Vector2<f32>,
    model: M4,
    view: M4,
    proj: M4,
}

impl UniformBufferObject {
    fn new(model: Similarity3<f32>, view: Isometry3<f32>, proj: Matrix4<f32>) -> Self {
        let flip_y = Similarity3::from_isometry(
            nalgebra::convert(Rotation3::from_euler_angles(
                std::f32::consts::PI,
                0.0,
                std::f32::consts::PI,
            )),
            -1.0,
        )
        .to_homogeneous();

        Self {
            _foo: Vector2::zeros(),
            model: M4(model.to_homogeneous()),
            view: M4(view.to_homogeneous()),
            proj: M4(flip_y * proj),
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        pos: Point3::new(-0.5, -0.5, 0.0),
        color: Color::RED,
    },
    Vertex {
        pos: Point3::new(0.5, -0.5, 0.0),
        color: Color::GREEN,
    },
    Vertex {
        pos: Point3::new(0.5, 0.5, 0.0),
        color: Color::BLUE,
    },
    Vertex {
        pos: Point3::new(-0.5, 0.5, 0.0),
        color: Color::WHITE,
    },
];
const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

#[derive(Debug)]
#[repr(transparent)]
struct Color([f32; 3]);
impl Color {
    const RED: Color = Color([1.0, 0.0, 0.0]);
    const GREEN: Color = Color([0.0, 1.0, 0.0]);
    const BLUE: Color = Color([0.0, 0.0, 1.0]);
    const WHITE: Color = Color([1.0, 1.0, 1.0]);
}

#[derive(Debug)]
struct Vertex {
    pos: Point3<f32>,
    color: Color,
}

impl Vertex {
    fn get_binding_descriptions() -> [VertexInputBindingDescription; 1] {
        [VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> [VertexInputAttributeDescription; 2] {
        [
            VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
        ]
    }
}

use std::ffi::{c_char, c_void, CStr};
use std::marker::PhantomData;

use ash::extensions::ext::DebugUtils;
use ash::vk::{
    DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerEXT, StructureType,
};
use ash::{vk, Entry, Instance};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

unsafe fn name_to_cstr(name: &[c_char]) -> &CStr {
    return CStr::from_ptr(name.as_ptr());
}

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
const EXTENSIONS: [&CStr; 0] = [];
#[cfg(all(target_os = "macos", not(debug_assertions)))]
const EXTENSIONS: [&CStr; 1] = [vk::KhrPortabilityEnumerationFn::name()];
#[cfg(all(target_os = "linux", debug_assertions))]
const EXTENSIONS: [&CStr; 1] = [vk::ExtDebugUtilsFn::name()];
#[cfg(all(target_os = "macos", debug_assertions))]
const EXTENSIONS: [&CStr; 2] = [
    vk::KhrPortabilityEnumerationFn::name(),
    vk::ExtDebugUtilsFn::name(),
];

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

struct Messenger<'a> {
    debug_utils: DebugUtils,
    messenger: DebugUtilsMessengerEXT,
    instance: PhantomData<&'a Instance>,
}

impl<'a> Messenger<'a> {
    unsafe extern "system" fn callback(
        message_severity: DebugUtilsMessageSeverityFlagsEXT,
        _message_type: DebugUtilsMessageTypeFlagsEXT,
        cb_data: *const DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut c_void,
    ) -> u32 {
        let message = CStr::from_ptr((*cb_data).p_message);
        eprintln!("validation callback: {message:?}");
        if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            panic!("Validation error");
        }
        vk::FALSE
    }

    fn new(entry: &'a Entry, instance: &'a Instance) -> Self {
        let debug_utils = DebugUtils::new(entry, instance);
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
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
            instance: PhantomData,
        }
    }
}

impl<'a> Drop for Messenger<'a> {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.messenger, None);
        }
    }
}

struct InstanceWrapper<'a> {
    instance: Instance,
    // debug_info: vk::DebugUtilsMessengerCreateInfoEXT,
    entry: PhantomData<&'a Entry>,
}

impl<'a> InstanceWrapper<'a> {
    fn new(entry: &'a Entry) -> Self {
        const APP_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"MyTest\0") };

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            p_application_name: APP_NAME.as_ptr(),
            s_type: StructureType::APPLICATION_INFO,
            ..Default::default()
        };

        let extensions = get_extensions(entry);
        let validation_layers = get_validation_layers(entry);

        #[cfg(debug_assertions)]
        let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
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

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            pp_enabled_extension_names: extensions.map(CStr::as_ptr).as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_layer_names: validation_layers.map(CStr::as_ptr).as_ptr(),
            enabled_layer_count: validation_layers.len() as u32,
            #[cfg(debug_assertions)]
            p_next: &debug_create_info as *const _ as *const c_void,
            #[cfg(target_os = "macos")]
            flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
            ..Default::default()
        };

        // let exts = entry.enumerate_instance_extension_properties(None).unwrap();

        Self {
            instance: unsafe { entry.create_instance(&create_info, None).unwrap() },
            entry: PhantomData,
        }
    }
}

impl<'a> Drop for InstanceWrapper<'a> {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();

    let entry = Entry::linked();
    let instance = InstanceWrapper::new(&entry);
    #[cfg(debug_assertions)]
    let _mes = Messenger::new(&entry, &instance.instance);

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

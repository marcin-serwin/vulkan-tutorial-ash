use std::ffi::{c_char, c_void, CStr};
use std::marker::PhantomData;

use ash::extensions::ext::DebugUtils;
use ash::vk::{
    DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT,
    StructureType,
};
use ash::{vk, Entry, Instance};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

unsafe fn name_to_cstr(name: &[c_char]) -> &CStr {
    return CStr::from_ptr(name.as_ptr());
}

#[cfg(not(debug_assertions))]
fn get_validation_layers(entry: &Entry) -> [*const c_char; 0] {
    []
}

#[cfg(debug_assertions)]
fn get_validation_layers(entry: &Entry) -> [*const c_char; 1] {
    let layers = [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
    let supported_layers = entry.enumerate_instance_layer_properties().unwrap();

    let names: Vec<&CStr> = supported_layers
        .iter()
        .map(|layer| unsafe { name_to_cstr(&layer.layer_name) })
        .collect();

    if layers.iter().any(|ext| !names.contains(ext)) {
        panic!("Required validation layer not found");
    }

    layers.map(|name| name.as_ptr())
}

fn get_extensions(entry: &Entry) -> Vec<*const c_char> {
    let mut extensions = vec![vk::KhrPortabilityEnumerationFn::name()];
    #[cfg(debug_assertions)]
    extensions.push(vk::ExtDebugUtilsFn::name());

    let supported_extensions = entry.enumerate_instance_extension_properties(None).unwrap();
    let names: Vec<&CStr> = supported_extensions
        .iter()
        .map(|ext| unsafe { name_to_cstr(&ext.extension_name) })
        .collect();

    if extensions.iter().any(|ext| !names.contains(ext)) {
        panic!("Required extension not found");
    }

    extensions.into_iter().map(|name| name.as_ptr()).collect()
}

struct Messenger<'a> {
    debug_utils: DebugUtils,
    messenger: DebugUtilsMessengerEXT,
    instance: PhantomData<&'a Instance>,
}

impl<'a> Messenger<'a> {
    unsafe extern "system" fn callback(
        _message_severity: DebugUtilsMessageSeverityFlagsEXT,
        _message_type: DebugUtilsMessageTypeFlagsEXT,
        cb_data: *const DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut c_void,
    ) -> u32 {
        let message = CStr::from_ptr((*cb_data).p_message);
        eprintln!("validation callback: {message:?}");
        if _message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
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
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            p_application_name: "MyTest\0".as_ptr() as *const c_char,
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
            pp_enabled_extension_names: extensions.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
            pp_enabled_layer_names: validation_layers.as_ptr(),
            enabled_layer_count: validation_layers.len() as u32,
            #[cfg(debug_assertions)]
            p_next: &debug_create_info as *const _ as *const c_void,
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
    let mes = Messenger::new(&entry, &instance.instance);

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

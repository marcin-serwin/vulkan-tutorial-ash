use std::ffi::{c_char, CStr};

use ash::vk::StructureType;
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

fn get_extensions(entry: &Entry) -> [*const c_char; 1] {
    let extensions = [vk::KhrPortabilityEnumerationFn::name()];
    let supported_extensions = entry.enumerate_instance_extension_properties(None).unwrap();
    let names: Vec<&CStr> = supported_extensions
        .iter()
        .map(|ext| unsafe { name_to_cstr(&ext.extension_name) })
        .collect();

    if extensions.iter().any(|ext| !names.contains(ext)) {
        panic!("Required extension not found");
    }

    extensions.map(|name| name.as_ptr())
}

fn create_instance(entry: &Entry) -> Instance {
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 0, 0),
        p_application_name: "MyTest\0".as_ptr() as *const c_char,
        s_type: StructureType::APPLICATION_INFO,
        ..Default::default()
    };

    let extensions = get_extensions(entry);
    let validation_layers = get_validation_layers(entry);

    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        pp_enabled_extension_names: extensions.as_ptr(),
        enabled_extension_count: extensions.len() as u32,
        flags: vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
        pp_enabled_layer_names: validation_layers.as_ptr(),
        enabled_layer_count: validation_layers.len() as u32,
        ..Default::default()
    };

    let exts = entry.enumerate_instance_extension_properties(None).unwrap();

    println!("Extensions: {exts:?}");
    unsafe { entry.create_instance(&create_info, None).unwrap() }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();

    let entry = Entry::linked();
    let instance = create_instance(&entry);

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

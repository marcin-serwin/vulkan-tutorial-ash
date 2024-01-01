use ash::{vk::SurfaceKHR, Entry, Instance};
#[cfg(target_os = "macos")]
use winit::raw_window_handle::AppKitWindowHandle;
use winit::raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle,
    WaylandWindowHandle,
};
use winit::window::Window;

pub fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> SurfaceKHR {
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
            create_wayland_surface(&entry, &instance, window, display)
        }
        #[cfg(target_os = "macos")]
        (RawWindowHandle::AppKit(window), RawDisplayHandle::AppKit(_)) => {
            create_app_kit_surface(&entry, &instance, window)
        }

        _ => panic!("unsupported windowing system"),
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

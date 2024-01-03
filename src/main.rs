use ash::Entry;
use std::sync::{atomic::AtomicU8, atomic::Ordering};
use vulkan_tutorial_ash::HelloTriangleApplication;
use winit::event::*;
use winit::keyboard::*;
use winit::{event_loop::EventLoop, window::Window};

fn main() {
    static FPS: AtomicU8 = AtomicU8::new(60);
    let event_loop = EventLoop::new().unwrap();
    let window = std::sync::Arc::new(Window::new(&event_loop).unwrap());
    let entry = Entry::linked();

    let mut app = HelloTriangleApplication::new(&entry, &window);

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    let win = std::sync::Arc::downgrade(&window);

    std::thread::spawn(move || {
        while let Some(window) = win.upgrade() {
            window.request_redraw();
            std::thread::sleep(std::time::Duration::from_secs_f64(
                1.0 / (FPS.load(Ordering::Relaxed) as f64),
            ));
        }
    });

    window
        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
        .unwrap();
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
                event: WindowEvent::Resized(size),
                ..
            } => {
                // println!("Resized {size:?}");
                app.occluded = size.width == 0 || size.height == 0;
                app.window_resized(size);
            }
            Event::WindowEvent {
                event: WindowEvent::Occluded(is_occluded),
                ..
            } => {
                println!("Occluded: {is_occluded:?}");
                app.occluded = is_occluded;
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                app.draw_frame();
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        event:
                            ref event @ KeyEvent {
                                physical_key: PhysicalKey::Code(code),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                app.handle_key_event(event);

                if event.state == ElementState::Pressed && !event.repeat {
                    use KeyCode::*;
                    match code {
                        Digit1 => {
                            FPS.store(10, Ordering::Relaxed);
                        }
                        Digit2 => {
                            FPS.store(30, Ordering::Relaxed);
                        }
                        Digit3 => {
                            FPS.store(60, Ordering::Relaxed);
                        }
                        Digit4 => {
                            FPS.store(144, Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                app.handle_mouse_movement(delta);
            }
            _ => {
                // println!("{event:?}");
            }
        })
        .unwrap();

    app.safe_drop()
        .map_err(|(_, e)| e)
        .expect("failed to drop the app");
}

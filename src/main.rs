use ash::Entry;
use vulkan_tutorial_ash::HelloTriangleApplication;
use winit::event::*;
use winit::keyboard::*;
use winit::{event_loop::EventLoop, window::Window};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();
    let entry = Entry::linked();

    let mut app = HelloTriangleApplication::new(&entry, &window);
    let mut fps: u8 = 60;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    window
        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
        .unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
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
                    fps = match code {
                        Digit1 => 10,
                        Digit2 => 30,
                        Digit3 => 60,
                        Digit4 => 144,
                        _ => fps,
                    }
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                app.handle_mouse_movement(delta);
            }
            Event::AboutToWait => {
                std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / (fps as f64)));
                window.request_redraw();
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

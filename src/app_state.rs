use std::time::Instant;

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use winit::event::ElementState;

pub struct AppState {
    pub view: Isometry3<f32>,
    pub model_rotation: f32,
    previous_draw: Instant,
    reversed: bool,
    keyboard_state: KeyboardState,
}

impl AppState {
    pub fn new(initial_view: Isometry3<f32>) -> Self {
        Self {
            view: initial_view,
            model_rotation: 0.0,
            reversed: false,
            previous_draw: Instant::now(),
            keyboard_state: KeyboardState::new(),
        }
    }

    pub fn get_view(&self) -> Isometry3<f32> {
        self.view
    }

    pub fn update_app_state(&mut self) {
        const SPEED: f32 = 5.0;

        let now = Instant::now();
        #[allow(clippy::cast_precision_loss)]
        let elapsed = now.duration_since(self.previous_draw).as_micros() as f32 / 1e6;
        self.previous_draw = now;
        self.model_rotation +=
            std::f32::consts::FRAC_PI_2 * elapsed * (if self.reversed { -1.0 } else { 1.0 });

        let moving_direction = SPEED
            * elapsed
            * ((self.keyboard_state.w.is_pressed() as u8 as f32) * Vector3::z()
                + (-1.0) * (self.keyboard_state.s.is_pressed() as u8 as f32) * Vector3::z()
                + (self.keyboard_state.a.is_pressed() as u8 as f32) * Vector3::x()
                + (-1.0) * (self.keyboard_state.d.is_pressed() as u8 as f32) * Vector3::x());

        self.view = Translation3::from(moving_direction) * self.view;
    }

    pub fn handle_key_event(&mut self, event: &winit::event::KeyEvent) {
        use winit::event::KeyEvent;
        use winit::keyboard::KeyCode::*;
        use winit::keyboard::PhysicalKey;
        let &KeyEvent {
            physical_key: PhysicalKey::Code(key_code),
            state,
            repeat,
            ..
        } = event
        else {
            return;
        };
        if repeat {
            return;
        }

        match key_code {
            KeyR if state == ElementState::Pressed => {
                self.reversed = !self.reversed;
            }
            KeyW => {
                self.keyboard_state.w = state;
            }
            KeyS => {
                self.keyboard_state.s = state;
            }
            KeyD => {
                self.keyboard_state.d = state;
            }
            KeyA => {
                self.keyboard_state.a = state;
            }
            _ => {}
        }
    }

    pub fn handle_mouse_movement(&mut self, delta: (f64, f64)) {
        const X_SENSITIVITY: f32 = 1.0 / 1000.0;
        const Y_SENSITIVITY: f32 = 1.0 / 1000.0;
        let pitch = delta.0 as f32 * X_SENSITIVITY;
        let roll = delta.1 as f32 * Y_SENSITIVITY;
        self.view =
            nalgebra::convert::<_, Isometry3<_>>(Rotation3::from_euler_angles(roll, pitch, 0.0))
                * self.view;
    }
}

#[derive(Debug)]
struct KeyboardState {
    w: ElementState,
    s: ElementState,
    a: ElementState,
    d: ElementState,
}

impl KeyboardState {
    fn new() -> Self {
        Self {
            w: ElementState::Released,
            s: ElementState::Released,
            a: ElementState::Released,
            d: ElementState::Released,
        }
    }
}

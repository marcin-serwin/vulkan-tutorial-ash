use std::env;
use std::process::Command;

fn get_ext(filename: &str) -> &str {
    let ext_start = filename.find('.').unwrap() + 1;
    &filename[ext_start..]
}

fn main() {
    const SHADERS_DIR: &str = "src/shaders";
    const SHADERS: [&str; 2] = ["shader.vert", "shader.frag"];
    if let Ok(path) = env::var("VULKAN_SDK_LIB_PATH") {
        println!(r"cargo:rustc-link-search={path}");
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let glslc_bin = env::var("GLSLC_BIN").unwrap_or("glslc".to_string());
    SHADERS.into_iter().for_each(|shader| {
        assert!(
            Command::new(glslc_bin.as_str())
                .args([
                    format!("{SHADERS_DIR}/{shader}").as_str(),
                    "-o",
                    format!("{}/{}.spv", out_dir, get_ext(shader)).as_str(),
                ])
                .status()
                .expect("failed to execute shader compilation command")
                .success(),
            "compilation of '{shader}' failed"
        );
    });

    println!(r"cargo:rerun-if-changed=src/shaders");
}

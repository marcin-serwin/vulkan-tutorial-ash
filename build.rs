use std::env;
use std::process::Command;

fn get_ext(filename: &str) -> &str {
    let ext_start = filename.find(".").unwrap() + 1;
    &filename[ext_start..]
}

fn main() {
    #[cfg(target_os = "macos")]
    println!(
        r"cargo:rustc-link-search={}",
        env::var("DYLD_FALLBACK_LIBRARY_PATH").unwrap()
    );

    let out_dir = env::var("OUT_DIR").unwrap();
    const SHADERS_DIR: &'static str = "src/shaders";
    const SHADERS: [&'static str; 2] = ["shader.vert", "shader.frag"];
    SHADERS.into_iter().for_each(|shader| {
        assert!(
            Command::new("glslc")
                .args(&[
                    format!("{}/{}", SHADERS_DIR, shader).as_str(),
                    "-o",
                    format!("{}/{}.spv", out_dir, get_ext(shader)).as_str(),
                ])
                .status()
                .unwrap()
                .success(),
            "compilation of '{shader}' failed"
        );
    });

    println!(r"cargo:rerun-if-changed=src/shaders");
}

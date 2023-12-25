fn main() {
    #[cfg(target_os = "macos")]
    println!(r"cargo:rustc-link-search=/Users/mserwin/VulkanSDK/1.3.268.1/macOS/lib");
}

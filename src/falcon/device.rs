use tch::Device;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn get_device() -> Device {
    Device::cuda_if_available()
}

#[cfg(target_os = "macos")]
pub fn get_device() -> Device {
    Device::Mps
}

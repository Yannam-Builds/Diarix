#[cfg(target_os = "macos")]
use std::process::Command;

fn main() {
    #[cfg(target_os = "macos")]
    {
        // ScreenCaptureKit is unavailable on macOS 11, so weak-link it and
        // gate capture support at runtime.
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,ScreenCaptureKit");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        println!("cargo:rustc-link-arg=-L/usr/lib/swift");

        if let Ok(output) = Command::new("xcode-select").arg("-p").output() {
            if output.status.success() {
                let xcode_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let swift_lib_path = format!(
                    "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx",
                    xcode_path
                );
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", swift_lib_path);
                println!("cargo:rustc-link-arg=-L{}", swift_lib_path);
            }
        }
    }

    // Tauri consumes the canonical Diarix icons from src-tauri/icons on every
    // platform. Keeping one icon source prevents platform-specific branding
    // from drifting again.
    tauri_build::build()
}

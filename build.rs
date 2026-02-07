use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/metal/ffi/metal_bridge.m");

    let metal_enabled = env::var("CARGO_FEATURE_METAL").is_ok();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !metal_enabled || target_os != "macos" {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    let obj_path = out_dir.join("metal_bridge.o");
    let lib_path = out_dir.join("libmetal_bridge.a");

    let clang_status = Command::new("xcrun")
        .args([
            "clang",
            "-fobjc-arc",
            "-fblocks",
            "-O2",
            "-c",
            "src/metal/ffi/metal_bridge.m",
            "-o",
        ])
        .arg(&obj_path)
        .status()
        .expect("failed to spawn xcrun clang");
    assert!(clang_status.success(), "xcrun clang failed");

    let ar_status = Command::new("ar")
        .args(["rcs"])
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .expect("failed to spawn ar");
    assert!(ar_status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=metal_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}

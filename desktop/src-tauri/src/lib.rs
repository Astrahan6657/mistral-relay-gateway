use std::env;
use std::fs::OpenOptions;
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, RunEvent, Url, WindowEvent,
};

struct CoreProcess(Mutex<Option<Child>>);

fn core_port() -> u16 {
    env::var("PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8790)
}

fn app_config_dir() -> PathBuf {
    if let Ok(home) = env::var("HOME").or_else(|_| env::var("USERPROFILE")) {
        PathBuf::from(home)
            .join(".config")
            .join("mistral-relay-gateway")
    } else {
        PathBuf::from(".")
    }
}

fn default_config_path() -> PathBuf {
    app_config_dir().join("config.json")
}

fn core_log_path() -> PathBuf {
    app_config_dir().join("core.log")
}

fn resolve_core_bin() -> PathBuf {
    if let Ok(custom) = env::var("MRG_CORE_BIN") {
        if !custom.trim().is_empty() {
            return PathBuf::from(custom);
        }
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sidecar_name = if cfg!(windows) {
                "mistral-relay-gateway.exe"
            } else {
                "mistral-relay-gateway"
            };
            let bundled = dir.join(sidecar_name);
            if bundled.is_file() {
                return bundled;
            }
        }
    }
    PathBuf::from("mistral-relay-gateway")
}

fn is_port_ready(port: u16) -> bool {
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

fn open_log_stdio() -> (Stdio, Stdio) {
    let log_path = core_log_path();
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(out_file) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
    {
        if let Ok(err_file) = out_file.try_clone() {
            return (Stdio::from(out_file), Stdio::from(err_file));
        }
    }
    (Stdio::null(), Stdio::null())
}

fn spawn_core() -> Option<Child> {
    let port = core_port();
    if is_port_ready(port) {
        return None;
    }

    let bin = resolve_core_bin();
    let cfg = env::var("CONFIG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_config_path());

    if let Some(parent) = cfg.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let (stdout_io, stderr_io) = open_log_stdio();

    Command::new(bin)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--config")
        .arg(cfg)
        .stdout(stdout_io)
        .stderr(stderr_io)
        .spawn()
        .ok()
}

fn stop_core(state: &CoreProcess) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut child) = guard.take() {
            #[cfg(unix)]
            {
                let pid = child.id().to_string();
                let _ = Command::new("kill").arg("-TERM").arg(&pid).status();
                thread::sleep(Duration::from_millis(300));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn open_log_file() {
    let path = core_log_path();
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(&path).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(&path).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("cmd")
            .args(["/C", "start", "", &path.to_string_lossy()])
            .spawn();
    }
}

fn navigate_when_ready(app_handle: tauri::AppHandle, port: u16) {
    thread::spawn(move || {
        for _ in 0..80 {
            if is_port_ready(port) {
                thread::sleep(Duration::from_millis(150));
                if let Some(win) = app_handle.get_webview_window("main") {
                    if let Ok(url) = Url::parse(&format!("http://127.0.0.1:{}/ui", port)) {
                        let _ = win.navigate(url);
                    }
                }
                return;
            }
            thread::sleep(Duration::from_millis(250));
        }
    });
}

pub fn run() {
    let port = core_port();
    let app = tauri::Builder::default()
        .manage(CoreProcess(Mutex::new(None)))
        .setup(move |app| {
            let child = spawn_core();
            if let Ok(mut guard) = app.state::<CoreProcess>().0.lock() {
                *guard = child;
            };

            navigate_when_ready(app.handle().clone(), port);

            let open_label = format!("打开控制台 (127.0.0.1:{})", port);
            let open_item = MenuItem::with_id(app, "open", &open_label, true, None::<&str>)?;
            let log_item = MenuItem::with_id(app, "logs", "查看核心日志 (core.log)", true, None::<&str>)?;
            let restart_item = MenuItem::with_id(app, "restart", "重启核心代理", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open_item, &log_item, &restart_item, &quit_item])?;

            let mut tray = TrayIconBuilder::new()
                .menu(&menu)
                .tooltip(format!("Mistral Relay Gateway (127.0.0.1:{})", port))
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "open" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "logs" => {
                        open_log_file();
                    }
                    "restart" => {
                        let state = app.state::<CoreProcess>();
                        stop_core(&state);
                        let new_child = spawn_core();
                        if let Ok(mut guard) = state.0.lock() {
                            *guard = new_child;
                        };
                        navigate_when_ready(app.clone(), port);
                    }
                    "quit" => {
                        let state = app.state::<CoreProcess>();
                        stop_core(&state);
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                });

            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to build Mistral Relay Desktop");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            let state = app_handle.state::<CoreProcess>();
            stop_core(&state);
        }
    });
}

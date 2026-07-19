use std::sync::Mutex;

use serde::Deserialize;
use tauri::image::Image;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::tray::{TrayIcon, TrayIconBuilder};
use tauri::{AppHandle, Emitter, Manager};

use crate::hotkey_monitor;
use crate::DICTATE_WINDOW_LABEL;

const TRAY_ICON_BYTES: &[u8] = include_bytes!("../icons/32x32.png");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrayState {
    Idle,
    Recording,
    Transcribing,
    Speaking,
    Error,
}

impl TrayState {
    fn from_frontend(value: &str) -> Result<Self, String> {
        match value {
            "idle" => Ok(Self::Idle),
            "recording" => Ok(Self::Recording),
            "transcribing" => Ok(Self::Transcribing),
            "speaking" => Ok(Self::Speaking),
            "error" => Ok(Self::Error),
            other => Err(format!("Unknown tray state '{other}'")),
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Self::Idle => "Diarix - ready for push-to-talk",
            Self::Recording => "Diarix - listening",
            Self::Transcribing => "Diarix - transcribing",
            Self::Speaking => "Diarix - speaking",
            Self::Error => "Diarix - action needed",
        }
    }
}

fn state_icon(state: TrayState) -> tauri::Result<Image<'static>> {
    let base = Image::from_bytes(TRAY_ICON_BYTES)?;
    if cfg!(target_os = "macos") || state == TrayState::Idle {
        return Ok(base);
    }

    // Keep the exact Diarix mark and only tint its visible white strokes.
    // This mirrors Handy's glanceable idle/listening/working tray states
    // without adding animated tray work or a second background process.
    let tint = match state {
        TrayState::Idle => [255, 255, 255],
        TrayState::Recording => [212, 172, 61],
        TrayState::Transcribing => [103, 149, 238],
        TrayState::Speaking => [160, 123, 210],
        TrayState::Error => [224, 92, 92],
    };
    let mut rgba = base.rgba().to_vec();
    for pixel in rgba.chunks_exact_mut(4) {
        if pixel[3] > 0 && pixel[0].max(pixel[1]).max(pixel[2]) > 72 {
            pixel[0] = tint[0];
            pixel[1] = tint[1];
            pixel[2] = tint[2];
        }
    }
    Ok(Image::new_owned(rgba, base.width(), base.height()))
}

pub struct CurrentTrayState(Mutex<TrayState>);

impl CurrentTrayState {
    fn new() -> Self {
        Self(Mutex::new(TrayState::Idle))
    }

    fn set(&self, state: TrayState) {
        *self.0.lock().unwrap() = state;
    }

    fn get(&self) -> TrayState {
        *self.0.lock().unwrap()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrayModel {
    model_name: String,
    display_name: String,
    loaded: bool,
}

#[derive(Clone, Debug, Default)]
struct TrayModelContext {
    models: Vec<TrayModel>,
    current_model: Option<String>,
}

pub struct CurrentTrayModels(Mutex<TrayModelContext>);

impl CurrentTrayModels {
    fn new() -> Self {
        Self(Mutex::new(TrayModelContext::default()))
    }

    fn get(&self) -> TrayModelContext {
        self.0.lock().unwrap().clone()
    }

    fn set(&self, models: Vec<TrayModel>, current_model: Option<String>) {
        *self.0.lock().unwrap() = TrayModelContext {
            models,
            current_model,
        };
    }
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn rebuild_menu(app: &AppHandle, state: TrayState) -> Result<(), String> {
    let status = MenuItem::with_id(app, "status", state.tooltip(), false, None::<&str>)
        .map_err(|error| error.to_string())?;
    let copy_last = MenuItem::with_id(
        app,
        "copy_last_transcript",
        "Copy last transcript",
        true,
        None::<&str>,
    )
    .map_err(|error| error.to_string())?;
    let open = MenuItem::with_id(app, "open", "Open Diarix", true, None::<&str>)
        .map_err(|error| error.to_string())?;
    let quit = MenuItem::with_id(app, "quit", "Quit Diarix", true, None::<&str>)
        .map_err(|error| error.to_string())?;
    let separator = || {
        PredefinedMenuItem::separator(app).map_err(|error| error.to_string())
    };

    let menu = match state {
        TrayState::Idle | TrayState::Error => {
            // Starting from a tray click is deliberately omitted. Opening the
            // menu makes the shell/taskbar the focused target, so a completed
            // dictation could paste there instead of the text field the user
            // intended. Handy follows the same split: the global shortcut
            // starts capture; the tray reflects state and offers cancellation.
            let context = app.state::<CurrentTrayModels>().get();
            let active = context
                .current_model
                .as_deref()
                .and_then(|current| {
                    context
                        .models
                        .iter()
                        .find(|model| model.model_name == current)
                });
            let model_label = active
                .map(|model| model.display_name.as_str())
                .unwrap_or("Speech model");
            let model_submenu =
                Submenu::with_id(app, "speech_model", model_label, !context.models.is_empty())
                    .map_err(|error| error.to_string())?;
            for model in &context.models {
                let item_id = format!("model_select:{}", model.model_name);
                let checked = context.current_model.as_deref() == Some(model.model_name.as_str());
                let item = CheckMenuItem::with_id(
                    app,
                    item_id,
                    &model.display_name,
                    true,
                    checked,
                    None::<&str>,
                )
                .map_err(|error| error.to_string())?;
                model_submenu
                    .append(&item)
                    .map_err(|error| error.to_string())?;
            }
            let unload = MenuItem::with_id(
                app,
                "unload_model",
                "Unload speech model",
                active.is_some_and(|model| model.loaded),
                None::<&str>,
            )
            .map_err(|error| error.to_string())?;
            Menu::with_items(
                app,
                &[
                    &status,
                    &separator()?,
                    &copy_last,
                    &separator()?,
                    &model_submenu,
                    &unload,
                    &separator()?,
                    &open,
                    &separator()?,
                    &quit,
                ],
            )
        }
        TrayState::Recording => {
            let stop = MenuItem::with_id(
                app,
                "stop",
                "Stop and transcribe",
                true,
                None::<&str>,
            )
            .map_err(|error| error.to_string())?;
            let cancel = MenuItem::with_id(
                app,
                "cancel",
                "Cancel dictation",
                true,
                None::<&str>,
            )
            .map_err(|error| error.to_string())?;
            Menu::with_items(
                app,
                &[
                    &status,
                    &separator()?,
                    &stop,
                    &cancel,
                    &separator()?,
                    &copy_last,
                    &separator()?,
                    &open,
                    &quit,
                ],
            )
        }
        TrayState::Transcribing => {
            let cancel = MenuItem::with_id(
                app,
                "cancel",
                "Cancel transcription",
                true,
                None::<&str>,
            )
            .map_err(|error| error.to_string())?;
            Menu::with_items(
                app,
                &[
                    &status,
                    &separator()?,
                    &cancel,
                    &separator()?,
                    &copy_last,
                    &separator()?,
                    &open,
                    &quit,
                ],
            )
        }
        TrayState::Speaking => {
            let stop = MenuItem::with_id(
                app,
                "stop_speaking",
                "Stop playback",
                true,
                None::<&str>,
            )
            .map_err(|error| error.to_string())?;
            Menu::with_items(
                app,
                &[
                    &status,
                    &separator()?,
                    &stop,
                    &separator()?,
                    &copy_last,
                    &separator()?,
                    &open,
                    &quit,
                ],
            )
        }
    }
    .map_err(|error| error.to_string())?;

    let tray = app.state::<TrayIcon>();
    tray.set_menu(Some(menu)).map_err(|error| error.to_string())?;
    tray.set_icon(Some(state_icon(state).map_err(|error| error.to_string())?))
        .map_err(|error| error.to_string())?;
    tray.set_tooltip(Some(state.tooltip()))
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub fn build(app: &AppHandle) -> tauri::Result<()> {
    app.manage(CurrentTrayState::new());
    app.manage(CurrentTrayModels::new());
    let icon = state_icon(TrayState::Idle)?;
    let tray = TrayIconBuilder::new()
        .icon(icon)
        .tooltip(TrayState::Idle.tooltip())
        .show_menu_on_left_click(true)
        .icon_as_template(cfg!(target_os = "macos"))
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => show_main_window(app),
            "stop" => hotkey_monitor::stop_dictation(app),
            "cancel" => hotkey_monitor::cancel_dictation(app),
            "stop_speaking" => {
                if let Some(window) = app.get_webview_window(DICTATE_WINDOW_LABEL) {
                    let _ = window.emit("dictate:speak-cancel", ());
                }
            }
            "copy_last_transcript" => {
                if let Some(window) = app.get_webview_window(DICTATE_WINDOW_LABEL) {
                    let _ = window.emit("dictate:copy-last-transcript", ());
                }
            }
            "unload_model" => {
                if let Some(window) = app.get_webview_window(DICTATE_WINDOW_LABEL) {
                    let _ = window.emit("dictate:unload-model", ());
                }
            }
            "quit" => app.exit(0),
            item if item.starts_with("model_select:") => {
                if let Some(window) = app.get_webview_window(DICTATE_WINDOW_LABEL) {
                    let model_name = item.trim_start_matches("model_select:");
                    let _ = window.emit(
                        "dictate:select-model",
                        serde_json::json!({ "modelName": model_name }),
                    );
                }
            }
            _ => {}
        })
        .build(app)?;
    app.manage(tray);
    rebuild_menu(app, TrayState::Idle)
        .map_err(|error| tauri::Error::Io(std::io::Error::other(error)))?;
    Ok(())
}

#[tauri::command]
pub fn set_tray_state(app: AppHandle, state: String) -> Result<(), String> {
    let state = TrayState::from_frontend(state.trim())?;
    app.state::<CurrentTrayState>().set(state);
    rebuild_menu(&app, state)
}

#[tauri::command]
pub fn set_tray_models(
    app: AppHandle,
    models: Vec<TrayModel>,
    current_model: Option<String>,
) -> Result<(), String> {
    app.state::<CurrentTrayModels>()
        .set(models, current_model);
    let state = app.state::<CurrentTrayState>().get();
    rebuild_menu(&app, state)
}

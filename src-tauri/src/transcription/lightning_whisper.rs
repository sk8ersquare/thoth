//! Lightning Whisper MLX backend with download progress support

use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;
use tauri::{AppHandle, Emitter};

static PYTHON_PATH: OnceLock<String> = OnceLock::new();

/// Get the directory where MLX models are stored: ~/.thoth/mlx_models/
pub fn get_mlx_models_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".thoth").join("mlx_models")
}

/// Find the Python executable that has lightning_whisper_mlx installed
fn find_python_with_lightning() -> &'static str {
    PYTHON_PATH.get_or_init(|| {
        let candidates = [
            "/usr/bin/python3",
            "python3",
            "python3.12",
            "python3.11",
            "python3.10",
            "python3.9",
            "python",
        ];
        for candidate in &candidates {
            let ok = Command::new(candidate)
                .args(["-c", "from lightning_whisper_mlx import LightningWhisperMLX"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                tracing::info!("Lightning Whisper MLX: found Python at {}", candidate);
                return candidate.to_string();
            }
        }
        tracing::warn!("Lightning Whisper MLX: no Python with package found, defaulting to /usr/bin/python3");
        "/usr/bin/python3".to_string()
    })
}

/// Map model+quant to expected directory name (mirrors lightning_whisper_mlx naming)
pub fn model_dir_name(model: &str, quant: Option<&str>) -> String {
    match quant {
        Some("4bit") if model.contains("distil") => format!("{}-4-bit", model),
        Some("8bit") if model.contains("distil") => format!("{}-8-bit", model),
        _ => model.to_string(),
    }
}

/// Check if a Lightning Whisper model is already downloaded
pub fn is_model_downloaded(model: &str, quant: Option<&str>) -> bool {
    let models_dir = get_mlx_models_dir();
    let dir_name = model_dir_name(model, quant);
    let model_path = models_dir.join(&dir_name);
    // Check for the two required files
    model_path.join("weights.npz").exists() && model_path.join("config.json").exists()
}

pub struct LightningWhisperTranscriptionService {
    model: String,
    quant: Option<String>,
}

impl LightningWhisperTranscriptionService {
    pub fn new(model: &str, quant: Option<&str>) -> Self {
        Self {
            model: model.to_string(),
            quant: quant.map(|q| q.to_string()),
        }
    }

    pub fn is_available() -> bool {
        let candidates = ["/usr/bin/python3", "python3"];
        for candidate in &candidates {
            let ok = Command::new(candidate)
                .args(["-c", "from lightning_whisper_mlx import LightningWhisperMLX"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                return true;
            }
        }
        false
    }

    pub fn transcribe(&self, audio_path: &Path) -> Result<String> {
        let quant_repr = match &self.quant {
            Some(q) => format!("\"{}\"", q),
            None => "None".to_string(),
        };

        // Script uses LightningWhisperMLX (not TranscriptionModel)
        // Sets local_dir context so model loads from ~/.thoth/mlx_models/
        let script = format!(
            r#"import sys, os; os.chdir(os.path.expanduser("~/.thoth")); from lightning_whisper_mlx import LightningWhisperMLX; m = LightningWhisperMLX(model="{model}", batch_size=12, quant={quant}); result = m.transcribe(sys.argv[1]); text = result.get("text", "") if isinstance(result, dict) else str(result); print(text.strip())"#,
            model = self.model,
            quant = quant_repr,
        );

        let audio_path_str = audio_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid audio path"))?;

        let python = find_python_with_lightning();

        tracing::info!(
            "Lightning Whisper MLX: transcribing {} model={} quant={:?} python={}",
            audio_path_str, self.model, self.quant, python,
        );

        let timeout = Duration::from_secs(300);
        let start = std::time::Instant::now();

        let mut child = Command::new(python)
            .args(["-c", &script, audio_path_str])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to start Lightning Whisper MLX: {}", e))?;

        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        return Err(anyhow!("Lightning Whisper MLX timed out after 300 seconds"));
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => return Err(anyhow!("Error waiting for Lightning Whisper MLX: {}", e)),
            }
        }

        let output = child.wait_with_output()
            .map_err(|e| anyhow!("Failed to read output: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "Lightning Whisper MLX failed (exit {}): {}",
                output.status,
                stderr.trim()
            ));
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        tracing::info!("Lightning Whisper MLX transcription complete: {} chars", text.len());
        Ok(text)
    }
}

// ── Tauri Commands ──────────────────────────────────────────────────────────

#[tauri::command]
pub fn is_lightning_whisper_available() -> bool {
    LightningWhisperTranscriptionService::is_available()
}

#[tauri::command]
pub fn check_lightning_model_downloaded(model: String, quant: Option<String>) -> bool {
    is_model_downloaded(&model, quant.as_deref())
}

#[tauri::command]
pub async fn install_lightning_whisper() -> Result<String, String> {
    let output = tokio::task::spawn_blocking(|| {
        std::process::Command::new("/usr/bin/pip3")
            .args(["install", "lightning-whisper-mlx"])
            .output()
            .or_else(|_| {
                std::process::Command::new("pip3")
                    .args(["install", "lightning-whisper-mlx"])
                    .output()
            })
    })
    .await
    .map_err(|e| format!("Task error: {}", e))?
    .map_err(|e| format!("Failed to run pip3: {}", e))?;

    if output.status.success() {
        Ok("lightning-whisper-mlx installed successfully".to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Installation failed: {}", stderr.trim()))
    }
}

/// Tauri command: Install lightning-whisper-mlx via pip in a terminal window.
///
/// Opens a new Terminal.app window (macOS) so the user can see install progress.
/// Returns immediately; the install runs in the background terminal.
#[tauri::command]
pub fn install_lightning_whisper_mlx() -> Result<(), String> {
    let script = "pip install lightning-whisper-mlx; echo ''; echo 'Done. You can close this window.'; read -p 'Press Enter to close...'";

    #[cfg(target_os = "macos")]
    {
        // Use osascript to open Terminal with the install command
        let osa = format!(
            r#"tell application "Terminal"
    do script "{}"
    activate
end tell"#,
            script.replace('"', "\\\"")
        );
        std::process::Command::new("osascript")
            .args(["-e", &osa])
            .spawn()
            .map_err(|e| format!("Failed to open Terminal: {}", e))?;
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Linux fallback: try x-terminal-emulator or xterm
        let tried = std::process::Command::new("x-terminal-emulator")
            .args(["-e", &format!("bash -c '{script}'")])
            .spawn();
        if tried.is_err() {
            std::process::Command::new("xterm")
                .args(["-e", &format!("bash -c '{script}'")])
                .spawn()
                .map_err(|e| format!("Failed to open terminal: {}", e))?;
        }
    }

    tracing::info!("Lightning Whisper MLX install triggered in external terminal");
    Ok(())
}

/// Download a Lightning Whisper model, emitting progress events to the frontend.
/// Emits `lightning-download-progress` events with { model, status, percent, message }.
/// Emits `lightning-download-complete` when done.
/// Emits `lightning-download-error` on failure.
#[tauri::command]
pub async fn download_lightning_model(
    app: AppHandle,
    model: String,
    quant: Option<String>,
) -> Result<(), String> {
    // Ensure model dir exists
    let models_dir = get_mlx_models_dir();
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Failed to create model dir: {}", e))?;

    let quant_repr = match quant.as_deref() {
        Some(q) => format!("\"{}\"", q),
        None => "None".to_string(),
    };

    // Python script that downloads model and prints progress to stdout
    // We chdir to ~/.thoth so the library's ./mlx_models/ resolves correctly
    let script = format!(
        r#"
import os, sys, json
os.chdir(os.path.expanduser("~/.thoth"))

from huggingface_hub import hf_hub_download
import lightning_whisper_mlx.lightning as lw

model_name = "{model}"
quant = {quant}

models = lw.models
if model_name not in models:
    print(json.dumps({{"error": f"Unknown model: {{model_name}}"}}))
    sys.exit(1)

if quant and "distil" not in model_name:
    repo_id = models[model_name].get(quant, models[model_name]["base"])
    dir_name = model_name
else:
    repo_id = models[model_name]["base"]
    dir_name = model_name
    if quant and "distil" in model_name:
        dir_name = model_name + ("-4-bit" if quant == "4bit" else "-8-bit")

if "distil" in model_name:
    files = [
        (f"./mlx_models/{{dir_name}}/weights.npz", "./"),
        (f"./mlx_models/{{dir_name}}/config.json", "./"),
    ]
else:
    files = [
        ("weights.npz", f"./mlx_models/{{dir_name}}"),
        ("config.json", f"./mlx_models/{{dir_name}}"),
    ]

total = len(files)
for i, (filename, local_dir) in enumerate(files):
    print(json.dumps({{"status": "downloading", "percent": int((i/total)*100), "message": "Downloading {} ({}/{})...".format(filename.split("/")[-1], i+1, total)}}), flush=True)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

print(json.dumps({{"status": "complete", "percent": 100, "message": "Download complete"}}), flush=True)
"#,
        model = model,
        quant = quant_repr,
    );

    let python = find_python_with_lightning();
    let model_clone = model.clone();

    let app_clone = app.clone();
    tokio::task::spawn_blocking(move || {
        let mut child = match Command::new(python)
            .args(["-c", &script])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = app_clone.emit("lightning-download-error", serde_json::json!({
                    "model": model_clone,
                    "error": format!("Failed to start Python: {}", e)
                }));
                return;
            }
        };

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let reader = BufReader::new(stdout);
        let mut stderr_lines = Vec::new();

        // Capture stderr in a background thread (non-blocking)
        let app_for_stderr = app_clone.clone();
        let stderr_handle = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                if let Ok(l) = line {
                    stderr_lines.push(l);
                }
            }
            stderr_lines
        });

        // Process stdout for progress events
        for line in reader.lines() {
            match line {
                Ok(l) => {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&l) {
                        if val.get("error").is_some() {
                            let _ = app_clone.emit("lightning-download-error", serde_json::json!({
                                "model": model_clone,
                                "error": val["error"].as_str().unwrap_or("Unknown error")
                            }));
                            return;
                        }
                        let _ = app_clone.emit("lightning-download-progress", serde_json::json!({
                            "model": model_clone,
                            "status": val["status"],
                            "percent": val["percent"],
                            "message": val["message"]
                        }));
                    }
                }
                Err(_) => break,
            }
        }

        let status = child.wait();
        match status {
            Ok(s) if !s.success() => {
                // Collect stderr from background thread
                let stderr_msg = stderr_handle.join().ok().map(|lines| {
                    lines.iter().rev().take(3).cloned().collect::<Vec<_>>().join("\n")
                }).filter(|s: &String| !s.is_empty());
                
                let error_msg = stderr_msg.unwrap_or_else(|| "Download process failed".to_string());
                let _ = app_clone.emit("lightning-download-error", serde_json::json!({
                    "model": model_clone,
                    "error": error_msg
                }));
            }
            Ok(_) => {
                let _ = app_clone.emit("lightning-download-complete", serde_json::json!({
                    "model": model_clone
                }));
            }
            Err(e) => {
                let _ = app_clone.emit("lightning-download-error", serde_json::json!({
                    "model": model_clone,
                    "error": format!("Failed to wait for process: {}", e)
                }));
            }
        }
    }).await.map_err(|e| format!("Task error: {}", e))?;

    Ok(())
}

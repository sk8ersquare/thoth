//! Lightning Whisper MLX backend
//!
//! Uses the lightning-whisper-mlx Python package for fast Whisper transcription
//! on Apple Silicon via MLX. Runs as a subprocess.
//!
//! The package exports `LightningWhisperMLX` (not `TranscriptionModel`).
//! Models are downloaded from HuggingFace on first use (cached in ./mlx_models/).
//! The package is typically installed under the system Python (/usr/bin/python3).

use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

/// Cache the discovered Python executable that has lightning_whisper_mlx
static PYTHON_PATH: OnceLock<String> = OnceLock::new();

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
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
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

    /// Check if lightning-whisper-mlx is importable with the correct class name
    pub fn is_available() -> bool {
        let candidates = ["/usr/bin/python3", "python3"];
        for candidate in &candidates {
            let ok = Command::new(candidate)
                .args(["-c", "from lightning_whisper_mlx import LightningWhisperMLX"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
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

        // Use LightningWhisperMLX (not TranscriptionModel which doesn't exist)
        // result may be a dict with "text" key or a plain string depending on version
        let script = format!(
            r#"from lightning_whisper_mlx import LightningWhisperMLX; import sys; m = LightningWhisperMLX(model="{model}", batch_size=12, quant={quant}); result = m.transcribe(sys.argv[1]); text = result.get("text", "") if isinstance(result, dict) else str(result); print(text.strip())"#,
            model = self.model,
            quant = quant_repr,
        );

        let audio_path_str = audio_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid audio path"))?;

        let python = find_python_with_lightning();

        tracing::info!(
            "Lightning Whisper MLX: transcribing {} with model={}, quant={:?}, python={}",
            audio_path_str,
            self.model,
            self.quant,
            python,
        );

        // Generous timeout — first run downloads model from HuggingFace
        let timeout = Duration::from_secs(300);
        let start = std::time::Instant::now();

        let mut child = Command::new(python)
            .args(["-c", &script, audio_path_str])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
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

        let output = child
            .wait_with_output()
            .map_err(|e| anyhow!("Failed to read Lightning Whisper MLX output: {}", e))?;

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

/// Tauri command: Check if Lightning Whisper MLX is available (correct class name)
#[tauri::command]
pub fn is_lightning_whisper_available() -> bool {
    LightningWhisperTranscriptionService::is_available()
}

/// Tauri command: Install lightning-whisper-mlx via pip3
#[tauri::command]
pub async fn install_lightning_whisper() -> Result<String, String> {
    // Install to the system python where it's most likely to work on macOS
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

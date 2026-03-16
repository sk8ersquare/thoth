//! Whisper MLX backend
//!
//! Uses the mlx-whisper Python package for fast Whisper transcription
//! on Apple Silicon via Apple's MLX framework. Runs as a Python subprocess.
//!
//! Models are downloaded from HuggingFace (mlx-community) on first use,
//! cached automatically by huggingface_hub to ~/.cache/huggingface/.
//!
//! Install: pip3 install mlx-whisper

use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

static PYTHON_PATH: OnceLock<String> = OnceLock::new();

/// Find Python with mlx_whisper installed
fn find_python() -> &'static str {
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
                .args(["-c", "import mlx_whisper"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                tracing::info!("MLX Whisper: found Python at {}", candidate);
                return candidate.to_string();
            }
        }
        tracing::warn!("MLX Whisper: no Python with mlx_whisper found, defaulting to /usr/bin/python3");
        "/usr/bin/python3".to_string()
    })
}

/// Check if mlx-whisper Python package is available
pub fn is_available() -> bool {
    Command::new("/usr/bin/python3")
        .args(["-c", "import mlx_whisper"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Map manifest model_id to the HuggingFace repo path for mlx-community
pub fn hf_repo_for_model(model_id: &str) -> &str {
    match model_id {
        "mlx-distil-large-v3"   => "mlx-community/distil-whisper-large-v3",
        "mlx-distil-medium-en"  => "mlx-community/distil-whisper-medium.en",
        "mlx-large-v3-turbo"    => "mlx-community/whisper-large-v3-turbo",
        "mlx-large-v3-turbo-q4" => "mlx-community/whisper-large-v3-turbo-q4",
        "mlx-large-v3"          => "mlx-community/whisper-large-v3-mlx",
        "mlx-small"             => "mlx-community/whisper-small-mlx",
        // Fallback: treat model_id as the HF repo directly
        other                   => other,
    }
}

pub struct MlxWhisperService {
    /// HuggingFace repo path, e.g. "mlx-community/whisper-large-v3-turbo"
    hf_repo: String,
}

impl MlxWhisperService {
    pub fn new(model_id: &str) -> Self {
        let hf_repo = hf_repo_for_model(model_id).to_string();
        tracing::info!("MLX Whisper: service created for repo={}", hf_repo);
        Self { hf_repo }
    }

    /// Pre-download the model weights from HuggingFace if not already cached.
    /// Called during init so the first transcription doesn't hang unexpectedly.
    pub fn ensure_cached(&self) {
        let python = find_python().to_string();
        let hf_repo = self.hf_repo.clone();
        // Run in background thread — don't block init
        std::thread::spawn(move || {
            tracing::info!("MLX Whisper: pre-fetching model weights for {} (background)", hf_repo);
            let script = format!(
                "from huggingface_hub import snapshot_download; \
                 snapshot_download(repo_id='{}', ignore_patterns=['*.md'])",
                hf_repo
            );
            let _ = Command::new(&python)
                .args(["-c", &script])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            tracing::info!("MLX Whisper: model weights ready for {}", hf_repo);
        });
    }

    pub fn transcribe(&self, audio_path: &Path) -> Result<String> {
        let audio_path_str = audio_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid audio path"))?;

        let python = find_python();
        let hf_repo = &self.hf_repo;

        // Load the WAV ourselves using Python's built-in `wave` module and pass
        // a numpy array directly to mlx_whisper — no ffmpeg dependency at all.
        // Thoth already records clean 16kHz mono PCM WAV files so no resampling needed.
        // Capture stdout around transcribe() to suppress mlx_whisper's own prints
        // (e.g. "Detected language: English"), then emit only the transcription text.
        let script = format!(
            "import wave, numpy as np, mlx_whisper, sys, io; \
             path = sys.argv[1]; \
             f = wave.open(path, 'rb'); \
             ch = f.getnchannels(); \
             frames = f.readframes(f.getnframes()); \
             f.close(); \
             audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0; \
             audio = audio.reshape(-1, ch).mean(axis=1) if ch > 1 else audio; \
             _old_stdout = sys.stdout; sys.stdout = io.StringIO(); \
             result = mlx_whisper.transcribe(audio, path_or_hf_repo='{}', verbose=False); \
             sys.stdout = _old_stdout; \
             print(result.get('text', '').strip() if isinstance(result, dict) else str(result).strip())",
            hf_repo
        );

        tracing::info!(
            "MLX Whisper: transcribing {} with repo={}",
            audio_path_str, hf_repo
        );

        let mut child = Command::new(python)
            .args(["-c", &script, audio_path_str])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to start mlx_whisper: {}", e))?;

        let timeout = Duration::from_secs(300);
        let start = std::time::Instant::now();

        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        return Err(anyhow!("MLX Whisper timed out after 300s"));
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => return Err(anyhow!("MLX Whisper wait error: {}", e)),
            }
        }

        let output = child.wait_with_output()
            .map_err(|e| anyhow!("Failed to read mlx_whisper output: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "MLX Whisper failed (exit {}): {}",
                output.status,
                stderr.trim()
            ));
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        tracing::info!("MLX Whisper: transcription complete ({} chars)", text.len());
        Ok(text)
    }
}

// ── Tauri Commands ────────────────────────────────────────────────────────────

#[tauri::command]
pub fn is_mlx_whisper_available() -> bool {
    is_available()
}

#[tauri::command]
pub fn install_mlx_whisper() -> Result<String, String> {
    let output = std::process::Command::new("/usr/bin/pip3")
        .args(["install", "mlx-whisper"])
        .output()
        .or_else(|_| {
            std::process::Command::new("pip3")
                .args(["install", "mlx-whisper"])
                .output()
        })
        .map_err(|e| format!("Failed to run pip3: {}", e))?;

    if output.status.success() {
        Ok("mlx-whisper installed successfully".to_string())
    } else {
        Err(format!(
            "Installation failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

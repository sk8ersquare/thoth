//! Whisper MLX backend
//!
//! Uses the mlx-whisper Python package for fast Whisper transcription
//! on Apple Silicon via Apple's MLX framework.
//!
//! The Python process is kept alive as a persistent daemon between transcriptions
//! so the model stays loaded in RAM — no per-call startup cost after warmup.
//!
//! Models are downloaded from HuggingFace (mlx-community) on first use,
//! cached automatically by huggingface_hub to ~/.cache/huggingface/.

use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, OnceLock};

static PYTHON_PATH: OnceLock<String> = OnceLock::new();

fn find_python() -> &'static str {
    PYTHON_PATH.get_or_init(|| {
        let candidates = [
            "/usr/bin/python3",
            "python3",
            "python3.12",
            "python3.11",
            "python3.10",
            "python3.9",
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

/// Map manifest model_id to the HuggingFace repo path
pub fn hf_repo_for_model(model_id: &str) -> &str {
    match model_id {
        "mlx-distil-large-v3"   => "mlx-community/distil-whisper-large-v3",
        "mlx-distil-medium-en"  => "mlx-community/distil-whisper-medium.en",
        "mlx-large-v3-turbo"    => "mlx-community/whisper-large-v3-turbo",
        "mlx-large-v3-turbo-q4" => "mlx-community/whisper-large-v3-turbo-q4",
        "mlx-large-v3"          => "mlx-community/whisper-large-v3-mlx",
        "mlx-small"             => "mlx-community/whisper-small-mlx",
        other                   => other,
    }
}

// ── Persistent Python daemon ──────────────────────────────────────────────────

/// The Python daemon script that stays alive, loads the model once,
/// then reads audio paths from stdin and writes transcription to stdout.
fn daemon_script(hf_repo: &str) -> String {
    format!(
        r#"
import sys, wave, io
import numpy as np
import mlx_whisper

# Suppress mlx_whisper's own stdout prints
import builtins
_real_print = builtins.print

HF_REPO = '{hf_repo}'
READY = 'READY'
DONE  = 'DONE'

# Pre-load model by running a tiny silent audio through it
_silence = np.zeros(3200, dtype=np.float32)
_old_stdout = sys.stdout; sys.stdout = io.StringIO()
try:
    mlx_whisper.transcribe(_silence, path_or_hf_repo=HF_REPO, verbose=False,
                            no_speech_threshold=0.9)
except Exception:
    pass
sys.stdout = _old_stdout

# Signal ready — Rust side waits for this line
_real_print(READY, flush=True)

# Main loop: read WAV paths, transcribe, print result + DONE sentinel
for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    try:
        with wave.open(path, 'rb') as f:
            ch     = f.getnchannels()
            frames = f.readframes(f.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            audio = audio.reshape(-1, ch).mean(axis=1)

        sys.stdout = io.StringIO()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=HF_REPO,
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        sys.stdout = sys.__stdout__
        text = result.get('text', '').strip() if isinstance(result, dict) else str(result).strip()
    except Exception as e:
        sys.stdout = sys.__stdout__
        text = ''

    _real_print(text, flush=True)
    _real_print(DONE, flush=True)
"#,
        hf_repo = hf_repo
    )
}

struct Daemon {
    _child:  Child,
    stdin:   ChildStdin,
    stdout:  BufReader<ChildStdout>,
}

pub struct MlxWhisperService {
    hf_repo: String,
    daemon:  Mutex<Option<Daemon>>,
}

impl MlxWhisperService {
    pub fn new(model_id: &str) -> Self {
        let hf_repo = hf_repo_for_model(model_id).to_string();
        tracing::info!("MLX Whisper: service created for repo={}", hf_repo);
        Self { hf_repo, daemon: Mutex::new(None) }
    }

    /// Spawn the Python daemon and wait for it to signal READY.
    /// Called once on first transcription (or after a crash restart).
    fn start_daemon(&self) -> Result<Daemon> {
        tracing::info!("MLX Whisper: starting daemon for {}", self.hf_repo);
        let script = daemon_script(&self.hf_repo);
        let python = find_python();

        let mut child = Command::new(python)
            .args(["-c", &script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn MLX Whisper daemon: {}", e))?;

        let stdin  = child.stdin.take()
            .ok_or_else(|| anyhow!("No stdin on MLX daemon"))?;
        let stdout = BufReader::new(
            child.stdout.take()
                .ok_or_else(|| anyhow!("No stdout on MLX daemon"))?
        );

        let mut daemon = Daemon { _child: child, stdin, stdout };

        // Wait for READY (model loaded) — can take 10-30s on first run (download + compile)
        tracing::info!("MLX Whisper: waiting for daemon READY (model loading/compiling)…");
        let mut line = String::new();
        loop {
            line.clear();
            let n = daemon.stdout.read_line(&mut line)
                .map_err(|e| anyhow!("MLX daemon stdout error: {}", e))?;
            if n == 0 {
                return Err(anyhow!("MLX daemon exited before READY"));
            }
            if line.trim() == "READY" {
                break;
            }
        }
        tracing::info!("MLX Whisper: daemon ready");
        Ok(daemon)
    }

    /// Pre-load: start daemon in background so it's warm before first recording.
    pub fn ensure_cached(&self) {
        // Kick the daemon off on a background thread
        let hf_repo = self.hf_repo.clone();
        std::thread::spawn(move || {
            tracing::info!("MLX Whisper: pre-warming daemon for {} (background)", hf_repo);
            let script = daemon_script(&hf_repo);
            let python = find_python();
            let mut child = match Command::new(python)
                .args(["-c", &script])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("MLX Whisper pre-warm failed to spawn: {}", e);
                    return;
                }
            };
            let stdout = match child.stdout.take() {
                Some(s) => BufReader::new(s),
                None => return,
            };
            let mut reader = stdout;
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) if line.trim() == "READY" => {
                        tracing::info!("MLX Whisper: pre-warm daemon ready, handing off to service");
                        break;
                    }
                    _ => {}
                }
            }
            // Daemon is now warmed up; it will be restarted on first actual transcription.
            // (We don't store this pre-warm daemon in self since we're in a different thread.)
            // Just let it exit — cost paid, model cached by mlx.
            let _ = child.kill();
        });
    }

    pub fn transcribe(&self, audio_path: &Path) -> Result<String> {
        let audio_path_str = audio_path.to_str()
            .ok_or_else(|| anyhow!("Invalid audio path"))?;

        let mut guard = self.daemon.lock()
            .map_err(|_| anyhow!("MLX daemon mutex poisoned"))?;

        // Start daemon if not running
        if guard.is_none() {
            *guard = Some(self.start_daemon()?);
        }

        let daemon = guard.as_mut().unwrap();

        // Send path, read back transcription + DONE sentinel
        writeln!(daemon.stdin, "{}", audio_path_str)
            .map_err(|e| anyhow!("MLX daemon stdin write failed: {}", e))?;

        let mut result_lines: Vec<String> = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            let n = daemon.stdout.read_line(&mut line)
                .map_err(|e| anyhow!("MLX daemon stdout read failed: {}", e))?;
            if n == 0 {
                // Daemon died — clear it so next call restarts
                *guard = None;
                return Err(anyhow!("MLX daemon exited unexpectedly"));
            }
            let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
            if trimmed == "DONE" {
                break;
            }
            result_lines.push(trimmed.to_string());
        }

        let text = result_lines.join(" ").trim().to_string();
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

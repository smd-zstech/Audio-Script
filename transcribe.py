#!/usr/bin/env python3
"""
Zscaler Training Audio Transcriber (Enhanced)
===============================================
Transcribes audio files to English text using OpenAI Whisper
with maximum accuracy optimizations for Zscaler training content.

Features:
    - Audio preprocessing (noise reduction, normalization)
    - Domain vocabulary biasing (Zscaler, SASE, SSE, etc.)
    - Beam search decoding with best-of sampling
    - Temperature fallback for difficult segments
    - Post-processing spell correction for domain terms
    - VAD-based segment filtering

Usage:
    python transcribe.py lecture.mp3                     # Default (medium model, max accuracy)
    python transcribe.py lecture.mp3 --model large       # Best possible accuracy
    python transcribe.py lecture.mp3 --timestamps        # Include timestamps
    python transcribe.py lecture.mp3 --fast              # Speed priority (base model, no preprocessing)
    python transcribe.py ./lectures/                     # Batch process directory

Models (accuracy vs speed):
    tiny    - Fastest, least accurate (~1GB VRAM)
    base    - Fast, decent accuracy (~1GB VRAM)
    small   - Good balance (~2GB VRAM)
    medium  - High accuracy (~5GB VRAM)  [DEFAULT]
    large   - Best accuracy (~10GB VRAM)
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# =============================================================================
# Zscaler & Network Security Domain Vocabulary
# =============================================================================

ZSCALER_TERMS = [
    # Zscaler Products & Services
    "Zscaler", "ZIA", "ZPA", "ZDX", "ZTNA", "Zscaler Internet Access",
    "Zscaler Private Access", "Zscaler Digital Experience",
    "Zscaler Client Connector", "Zscaler Cloud Connector",
    "Zscaler Branch Connector", "Zscaler Workload Communications",
    "Zscaler Deception", "Zscaler Data Protection",
    "Zscaler Zero Trust Exchange", "ZTX",
    "Zscaler Cloud Firewall", "Zscaler Sandbox",
    "Zscaler Browser Isolation", "Cloud Browser Isolation", "CBI",
    "Zscaler CASB", "Zscaler DLP",
    # Zscaler Specific Concepts
    "App Connector", "Service Edge", "Private Service Edge", "PSE",
    "Cloud Connector", "Branch Connector",
    "Nanolog", "Nanolog Streaming Service", "NSS",
    "Z-Tunnel", "Tunnel 1.0", "Tunnel 2.0", "Tunnel with Local Proxy",
    "PAC file", "GRE tunnel", "IPsec tunnel",
    "Zscaler Enforcement Node", "ZEN",
    "Central Authority", "CA",
    "Policy Engine",
]

NETWORK_SECURITY_TERMS = [
    # SASE / SSE / Zero Trust
    "SASE", "Secure Access Service Edge",
    "SSE", "Security Service Edge",
    "Zero Trust", "ZTNA", "Zero Trust Network Access",
    "SDP", "Software Defined Perimeter",
    "SWG", "Secure Web Gateway",
    "CASB", "Cloud Access Security Broker",
    "DLP", "Data Loss Prevention", "Data Leakage Prevention",
    "FWaaS", "Firewall as a Service",
    "RBI", "Remote Browser Isolation",
    # Network Fundamentals
    "MPLS", "SD-WAN", "VPN", "IPsec", "GRE",
    "BGP", "OSPF", "DNS", "DHCP", "NAT", "PAT",
    "TCP", "UDP", "HTTP", "HTTPS", "TLS", "SSL",
    "ICMP", "ARP", "VLAN", "VXLAN",
    "QoS", "Quality of Service",
    "MTU", "MSS", "RTT", "latency", "jitter", "throughput",
    "bandwidth", "packet loss",
    "proxy", "forward proxy", "reverse proxy", "explicit proxy", "transparent proxy",
    "load balancer", "CDN", "Content Delivery Network",
    "peering", "transit", "IX", "Internet Exchange",
    # Security Concepts
    "firewall", "IDS", "IPS", "Intrusion Detection System", "Intrusion Prevention System",
    "NGFW", "Next-Generation Firewall",
    "WAF", "Web Application Firewall",
    "SIEM", "Security Information and Event Management",
    "SOAR", "SOC", "Security Operations Center",
    "EDR", "Endpoint Detection and Response",
    "XDR", "Extended Detection and Response",
    "MDR", "Managed Detection and Response",
    "IAM", "Identity and Access Management",
    "MFA", "Multi-Factor Authentication", "SSO", "Single Sign-On",
    "SAML", "OAuth", "OIDC", "OpenID Connect", "IdP", "Identity Provider",
    "RBAC", "Role-Based Access Control",
    "PKI", "Public Key Infrastructure",
    "certificate", "CA certificate", "root certificate",
    # Threat & Compliance
    "malware", "ransomware", "phishing", "spear phishing",
    "C2", "command and control", "botnet",
    "APT", "Advanced Persistent Threat",
    "CVE", "vulnerability", "exploit", "zero-day",
    "OWASP", "MITRE ATT&CK",
    "compliance", "GDPR", "HIPAA", "PCI DSS", "SOC 2", "ISO 27001",
    "encryption", "decryption", "SSL inspection", "TLS inspection",
    "sandboxing", "URL filtering", "content filtering",
    # Cloud & Infrastructure
    "IaaS", "PaaS", "SaaS",
    "AWS", "Azure", "GCP", "Google Cloud Platform",
    "multi-cloud", "hybrid cloud",
    "container", "Kubernetes", "Docker",
    "API", "REST API", "microservices",
    "CI/CD", "DevOps", "DevSecOps",
]

_ALL_TERMS = ZSCALER_TERMS + NETWORK_SECURITY_TERMS
DOMAIN_PROMPT = (
    "This is a Zscaler cybersecurity training lecture in English. "
    "Key terms: " + ", ".join(_ALL_TERMS) + "."
)

# =============================================================================
# Post-processing: Domain term spell correction
# =============================================================================
# Maps common Whisper misrecognitions -> correct domain term
# Case-insensitive matching, preserves surrounding context

SPELL_CORRECTIONS = {
    # Zscaler misrecognitions
    r"\bz\s*scaler\b": "Zscaler",
    r"\bzee\s*scaler\b": "Zscaler",
    r"\bz-?scalar\b": "Zscaler",
    r"\bzia\b": "ZIA",
    r"\bzpa\b": "ZPA",
    r"\bzdx\b": "ZDX",
    r"\bztna\b": "ZTNA",
    r"\bz\s*tunnel\b": "Z-Tunnel",
    r"\bnano\s*log\b": "Nanolog",
    r"\bzen\b(?!\s+(?:garden|buddhis|meditation))": "ZEN",
    # SASE/SSE misrecognitions
    r"\bsassy\b": "SASE",
    r"\bsassey\b": "SASE",
    r"\bsase\b": "SASE",
    r"\bsse\b": "SSE",
    # Network terms
    r"\bsd\s*wan\b": "SD-WAN",
    r"\bsd\s*-\s*one\b": "SD-WAN",
    r"\bmpls\b": "MPLS",
    r"\bi\s*p\s*sec\b": "IPsec",
    r"\bbgp\b": "BGP",
    r"\bospf\b": "OSPF",
    r"\bvlan\b": "VLAN",
    r"\bvxlan\b": "VXLAN",
    r"\bqos\b": "QoS",
    # Security terms
    r"\bswg\b": "SWG",
    r"\bcasb\b": "CASB",
    r"\bdlp\b": "DLP",
    r"\bngfw\b": "NGFW",
    r"\bwaf\b": "WAF",
    r"\bsiem\b": "SIEM",
    r"\bsoar\b": "SOAR",
    r"\bedr\b": "EDR",
    r"\bxdr\b": "XDR",
    r"\bmdr\b": "MDR",
    r"\biam\b": "IAM",
    r"\bmfa\b": "MFA",
    r"\bsso\b": "SSO",
    r"\bsaml\b": "SAML",
    r"\boidc\b": "OIDC",
    r"\bidp\b": "IdP",
    r"\brbac\b": "RBAC",
    r"\bpki\b": "PKI",
    r"\bapt\b(?!\s+(?:get|install|update))": "APT",
    r"\bcve\b": "CVE",
    r"\bowasp\b": "OWASP",
    r"\bmitre\b": "MITRE",
    # Cloud terms
    r"\biaas\b": "IaaS",
    r"\bpaas\b": "PaaS",
    r"\bsaas\b": "SaaS",
    r"\bgcp\b": "GCP",
    r"\bci\s*/\s*cd\b": "CI/CD",
    r"\bdev\s*sec\s*ops\b": "DevSecOps",
    # FWaaS
    r"\bfw\s*aas\b": "FWaaS",
    r"\bfire\s*wall\s*as\s*a\s*service\b": "Firewall as a Service",
    # RBI / CBI
    r"\brbi\b": "RBI",
    r"\bcbi\b": "CBI",
}


def apply_spell_corrections(text: str) -> str:
    """Apply domain-specific spell corrections to transcribed text."""
    for pattern, replacement in SPELL_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# =============================================================================
# Audio Preprocessing
# =============================================================================

SUPPORTED_FORMATS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".wma",
    ".aac", ".opus", ".webm", ".mpeg", ".mpga",
}


def preprocess_audio(audio_path: str, verbose: bool = True) -> str:
    """
    Preprocess audio for better recognition:
    1. Load and convert to mono 16kHz WAV (Whisper's native format)
    2. Apply noise reduction
    3. Normalize volume
    Returns path to preprocessed temp file.
    """
    try:
        import numpy as np
        import noisereduce as nr
        import whisper
    except ImportError as e:
        if verbose:
            print(f"  Preprocessing skipped (missing: {e.name}). Using raw audio.")
        return audio_path

    if verbose:
        print("  [1/3] Loading audio...")

    # Load audio at Whisper's native 16kHz sample rate
    audio = whisper.load_audio(audio_path)

    if verbose:
        duration = len(audio) / 16000
        print(f"        Duration: {duration:.1f}s")
        print("  [2/3] Reducing noise...")

    # Noise reduction - estimate noise from quietest parts
    audio_denoised = nr.reduce_noise(
        y=audio,
        sr=16000,
        prop_decrease=0.6,      # moderate noise reduction (not too aggressive)
        n_fft=2048,
        stationary=True,        # good for consistent background noise (AC, fans, etc.)
    )

    if verbose:
        print("  [3/3] Normalizing volume...")

    # Peak normalization to -1dB to prevent clipping
    peak = np.max(np.abs(audio_denoised))
    if peak > 0:
        target_peak = 10 ** (-1.0 / 20)  # -1dB
        audio_denoised = audio_denoised * (target_peak / peak)

    # Save preprocessed audio as temp file
    import tempfile
    import scipy.io.wavfile

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"zscaler_preprocessed_{os.path.basename(audio_path)}.wav"
    )
    # Convert float32 [-1,1] to int16
    audio_int16 = (audio_denoised * 32767).astype(np.int16)
    scipy.io.wavfile.write(temp_path, 16000, audio_int16)

    if verbose:
        print("  Preprocessing complete.")

    return temp_path


# =============================================================================
# Core Transcription
# =============================================================================

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_audio_files(path: str) -> list[str]:
    """Get list of audio files from a file path or directory."""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_FORMATS:
            return [str(p)]
        else:
            print(f"Error: Unsupported format '{p.suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
            sys.exit(1)
    elif p.is_dir():
        files = []
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
                files.append(str(f))
        if not files:
            print(f"Error: No audio files found in '{path}'")
            sys.exit(1)
        return files
    else:
        print(f"Error: '{path}' does not exist")
        sys.exit(1)


def transcribe_file(
    model,
    audio_path: str,
    task: str = "transcribe",
    language: str | None = None,
    include_timestamps: bool = False,
    verbose: bool = True,
    fast_mode: bool = False,
) -> dict:
    """Transcribe a single audio file with maximum accuracy settings."""
    filename = os.path.basename(audio_path)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"Task: {'Translate to English' if task == 'translate' else 'Transcribe'}")
        print(f"Mode: {'Fast' if fast_mode else 'Max Accuracy'}")
        print(f"{'='*60}")

    # --- Audio Preprocessing ---
    if not fast_mode:
        processed_path = preprocess_audio(audio_path, verbose=verbose)
    else:
        processed_path = audio_path

    start_time = time.time()

    # --- Whisper Decode Options ---
    options = {
        "task": task,
        "fp16": False,
        "initial_prompt": DOMAIN_PROMPT,
    }

    if language:
        options["language"] = language
    else:
        options["language"] = "en"

    if not fast_mode:
        # === MAX ACCURACY SETTINGS ===

        # Beam search: explore more paths for better decoding
        options["beam_size"] = 10

        # Best-of: sample N candidates, pick highest probability
        options["best_of"] = 10

        # Temperature fallback: start greedy (0.0), fall back to sampling
        # if compression_ratio or logprob thresholds are not met
        options["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

        # Compression ratio threshold: reject segments that are too repetitive
        # (lower = stricter, default 2.4)
        options["compression_ratio_threshold"] = 2.0

        # Log probability threshold: reject low-confidence segments
        # (higher = stricter, default -1.0)
        options["logprob_threshold"] = -0.5

        # No-speech threshold: skip segments that are likely silence
        # (lower = more aggressive skip, default 0.6)
        options["no_speech_threshold"] = 0.4

        # Condition on previous text: use prior segment as context
        # Helps maintain consistency across segments
        options["condition_on_previous_text"] = True

        # Word-level timestamps for finer granularity
        options["word_timestamps"] = True

    if verbose:
        print(f"\n  Transcribing with Whisper...")

    result = model.transcribe(processed_path, **options)

    elapsed = time.time() - start_time

    # --- Post-processing: Domain spell correction ---
    if not fast_mode:
        if verbose:
            print("  Applying domain spell corrections...")
        result["text"] = apply_spell_corrections(result["text"])
        for seg in result.get("segments", []):
            seg["text"] = apply_spell_corrections(seg["text"])

    # --- Clean up temp file ---
    if processed_path != audio_path and os.path.exists(processed_path):
        os.remove(processed_path)

    if verbose:
        detected_lang = result.get("language", "unknown")
        print(f"  Detected language: {detected_lang}")
        print(f"  Processing time: {elapsed:.1f}s")

    return result


def format_output(result: dict, include_timestamps: bool = False) -> str:
    """Format transcription result as text."""
    if not include_timestamps:
        return result["text"].strip()

    lines = []
    for segment in result.get("segments", []):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        lines.append(f"[{start} -> {end}]  {text}")
    return "\n".join(lines)


def save_output(text: str, output_path: str):
    """Save transcription to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Zscaler Training Audio Transcriber (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s lecture.mp3                              # Default: medium model, max accuracy
  %(prog)s lecture.mp3 --model large                # Best accuracy (needs ~10GB VRAM)
  %(prog)s lecture.mp3 --timestamps                 # Include timestamps
  %(prog)s lecture.mp3 --fast                       # Fast mode (skip preprocessing)
  %(prog)s lecture.mp3 --language ko --translate    # Korean audio -> English
  %(prog)s ./lectures/ -o all_transcripts.txt       # Batch process directory
        """,
    )
    parser.add_argument("input", help="Audio file or directory containing audio files")
    parser.add_argument("-o", "--output", help="Output file path (default: save as .txt next to audio)")
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate non-English audio to English",
    )
    parser.add_argument(
        "--language",
        help="Source language code (e.g., ko, ja, zh, es). Default: en",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in the output",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip preprocessing, use greedy decoding (less accurate but faster)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # --- Import check ---
    try:
        import whisper
    except ImportError:
        print("Error: OpenAI Whisper is not installed.")
        print("Install with: pip install openai-whisper")
        sys.exit(1)

    if not args.fast:
        missing = []
        try:
            import noisereduce  # noqa: F401
        except ImportError:
            missing.append("noisereduce")
        try:
            import scipy  # noqa: F401
        except ImportError:
            missing.append("scipy")
        if missing:
            print(f"Warning: {', '.join(missing)} not installed. Audio preprocessing disabled.")
            print(f"Install with: pip install {' '.join(missing)}")
            print()

    audio_files = get_audio_files(args.input)
    task = "translate" if args.translate else "transcribe"

    if not args.quiet:
        print(f"Loading Whisper model '{args.model}'...")
        if not args.fast:
            print("Mode: MAX ACCURACY (beam=10, best_of=10, noise reduction, spell correction)")
        else:
            print("Mode: FAST (greedy decoding, no preprocessing)")

    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nIf this is your first run, the model needs to be downloaded.")
        print("Make sure you have internet access and try again.")
        print(f"\nModels are cached at: ~/.cache/whisper/")
        sys.exit(1)

    if not args.quiet:
        print(f"Model loaded. Processing {len(audio_files)} file(s)...")

    all_results = []
    for audio_path in audio_files:
        result = transcribe_file(
            model=model,
            audio_path=audio_path,
            task=task,
            language=args.language,
            include_timestamps=args.timestamps,
            verbose=not args.quiet,
            fast_mode=args.fast,
        )
        text = format_output(result, include_timestamps=args.timestamps)
        all_results.append((audio_path, text))

    # --- Output results ---
    if len(all_results) == 1:
        audio_path, text = all_results[0]
        print(f"\n{'='*60}")
        print("TRANSCRIPTION RESULT")
        print(f"{'='*60}")
        print(text)
        print(f"{'='*60}\n")

        output_path = args.output or Path(audio_path).with_suffix(".txt")
        save_output(text, str(output_path))
    else:
        combined = []
        for audio_path, text in all_results:
            filename = os.path.basename(audio_path)
            header = f"\n{'='*60}\n[{filename}]\n{'='*60}"
            combined.append(f"{header}\n{text}")
            print(header)
            print(text)

            individual_output = Path(audio_path).with_suffix(".txt")
            save_output(text, str(individual_output))

        if args.output:
            full_text = "\n\n".join(combined)
            save_output(full_text, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()

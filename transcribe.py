#!/usr/bin/env python3
"""
Zscaler Training Audio Transcriber
===================================
Transcribes audio files to English text using OpenAI Whisper.
Supports translation from any language to English.

Usage:
    python3 transcribe.py <audio_file>                    # Transcribe a single file
    python3 transcribe.py <audio_file> -o output.txt      # Save to specific output file
    python3 transcribe.py <audio_dir>                     # Transcribe all audio files in directory
    python3 transcribe.py <audio_file> --model medium      # Use a specific model size
    python3 transcribe.py <audio_file> --translate         # Translate non-English audio to English
    python3 transcribe.py <audio_file> --timestamps        # Include timestamps in output

Models (accuracy vs speed tradeoff):
    tiny    - Fastest, least accurate (~1GB VRAM)
    base    - Fast, decent accuracy (~1GB VRAM)
    small   - Good balance (~2GB VRAM)
    medium  - High accuracy (~5GB VRAM)
    large   - Best accuracy (~10GB VRAM)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# =============================================================================
# Zscaler & Network Security Domain Vocabulary
# =============================================================================
# Whisper uses initial_prompt to bias recognition toward these terms.
# Grouped by category for maintainability.

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

# Build the prompt string Whisper will use for vocabulary biasing
_ALL_TERMS = ZSCALER_TERMS + NETWORK_SECURITY_TERMS
DOMAIN_PROMPT = (
    "This is a Zscaler cybersecurity training lecture in English. "
    "Key terms: " + ", ".join(_ALL_TERMS) + "."
)

SUPPORTED_FORMATS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".wma",
    ".aac", ".opus", ".webm", ".mpeg", ".mpga",
}


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
) -> dict:
    """Transcribe or translate a single audio file."""
    filename = os.path.basename(audio_path)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"Task: {'Translate to English' if task == 'translate' else 'Transcribe'}")
        print(f"{'='*60}")

    start_time = time.time()

    options = {
        "task": task,
        "fp16": False,
        "initial_prompt": DOMAIN_PROMPT,
    }
    if language:
        options["language"] = language
    else:
        # Default to English for Zscaler training content
        options["language"] = "en"

    result = model.transcribe(audio_path, **options)

    elapsed = time.time() - start_time

    if verbose:
        detected_lang = result.get("language", "unknown")
        print(f"Detected language: {detected_lang}")
        print(f"Processing time: {elapsed:.1f}s")

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


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to English text (Zscaler Training Transcriber)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recording.mp3                          # Basic transcription
  %(prog)s recording.mp3 --translate              # Translate to English
  %(prog)s recording.mp3 --timestamps             # Include timestamps
  %(prog)s recording.mp3 --model medium            # Use medium model
  %(prog)s ./recordings/ --translate -o output.txt # Batch translate directory
  %(prog)s recording.mp3 --language ko --translate # Korean audio -> English
        """,
    )
    parser.add_argument("input", help="Audio file or directory containing audio files")
    parser.add_argument("-o", "--output", help="Output file path (default: print to stdout and save as .txt)")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate non-English audio to English (instead of just transcribing)",
    )
    parser.add_argument(
        "--language",
        help="Source language code (e.g., ko, ja, zh, es). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in the output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Lazy import to show helpful error if whisper not installed
    try:
        import whisper
    except ImportError:
        print("Error: OpenAI Whisper is not installed.")
        print("Install it with: pip3 install openai-whisper")
        sys.exit(1)

    audio_files = get_audio_files(args.input)
    task = "translate" if args.translate else "transcribe"

    if not args.quiet:
        print(f"Loading Whisper model '{args.model}'...")

    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nIf this is your first run, the model needs to be downloaded.")
        print("Make sure you have internet access and try again.")
        print(f"\nYou can also manually download models to: ~/.cache/whisper/")
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
        )
        text = format_output(result, include_timestamps=args.timestamps)
        all_results.append((audio_path, text))

    # Output results
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
        # Multiple files
        combined = []
        for audio_path, text in all_results:
            filename = os.path.basename(audio_path)
            header = f"\n{'='*60}\n[{filename}]\n{'='*60}"
            combined.append(f"{header}\n{text}")
            print(header)
            print(text)

            # Save individual file
            individual_output = Path(audio_path).with_suffix(".txt")
            save_output(text, str(individual_output))

        # Save combined output if specified
        if args.output:
            full_text = "\n\n".join(combined)
            save_output(full_text, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()

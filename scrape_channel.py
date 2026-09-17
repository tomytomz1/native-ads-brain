#!/usr/bin/env python3
"""Export public YouTube captions for every video on a channel.

Uses yt-dlp to list videos and youtube-transcript-api to read the same
caption track YouTube shows via "Show transcript" (plain + timestamped).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig, ProxyConfig, WebshareProxyConfig
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)

try:
    from youtube_transcript_api._errors import PoTokenRequired
except ImportError:  # older library versions
    PoTokenRequired = type("PoTokenRequired", (CouldNotRetrieveTranscript,), {})

LOG = logging.getLogger("scrape_channel")

DEFAULT_CHANNEL = "https://www.youtube.com/@revenuetactics/videos"
PREFERRED_LANGS = ("en", "en-US", "en-GB", "en-CA", "en-AU")
RETRYABLE = (
    IpBlocked,
    RequestBlocked,
    YouTubeRequestFailed,
    ConnectionError,
    TimeoutError,
    OSError,
)


class NoTranscriptAvailable(Exception):
    """Raised when a video has no caption tracks at all."""


class IpStillBlocked(Exception):
    """Raised when YouTube keeps blocking caption requests from this IP."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public YouTube captions for every video on a channel."
    )
    parser.add_argument(
        "channel_url",
        nargs="?",
        default=DEFAULT_CHANNEL,
        help=f"Channel URL (default: {DEFAULT_CHANNEL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: transcripts/<channel-name>/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
        help="Seconds to wait between transcript requests (default: 2.5)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Retries per video on transient request failures (default: 4)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=120.0,
        help="Seconds to wait after a YouTube IP block before retrying (default: 120)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Pause after this many newly saved videos (default: 12)",
    )
    parser.add_argument(
        "--batch-pause",
        type=float,
        default=20.0,
        help="Seconds to pause between batches (default: 20)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Stop after listing this many videos (for testing)",
    )
    parser.add_argument(
        "--languages",
        default="en,en-US,en-GB",
        help="Comma-separated language preference (default: en,en-US,en-GB)",
    )
    parser.add_argument(
        "--proxy",
        default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "",
        help="HTTP(S)/SOCKS proxy URL, e.g. http://user:pass@host:port",
    )
    parser.add_argument(
        "--webshare-username",
        default=os.environ.get("WEBSHARE_USERNAME", ""),
        help="Webshare rotating-residential username (or env WEBSHARE_USERNAME)",
    )
    parser.add_argument(
        "--webshare-password",
        default=os.environ.get("WEBSHARE_PASSWORD", ""),
        help="Webshare rotating-residential password (or env WEBSHARE_PASSWORD)",
    )
    return parser.parse_args()


def build_proxy_config(args: argparse.Namespace) -> ProxyConfig | None:
    username = (args.webshare_username or "").strip()
    password = (args.webshare_password or "").strip()
    if username and password:
        LOG.info("Using Webshare rotating residential proxies")
        return WebshareProxyConfig(
            proxy_username=username,
            proxy_password=password,
            retries_when_blocked=10,
        )
    if username or password:
        raise SystemExit(
            "Both --webshare-username and --webshare-password are required "
            "(or WEBSHARE_USERNAME and WEBSHARE_PASSWORD)."
        )
    proxy = (args.proxy or "").strip()
    if proxy:
        LOG.info("Using HTTP proxy %s", urlparse(proxy).hostname or proxy)
        return GenericProxyConfig(http_url=proxy, https_url=proxy)
    return None


def channel_slug(channel_url: str) -> str:
    path = urlparse(channel_url).path.strip("/")
    parts = [p for p in path.split("/") if p]
    for part in parts:
        if part.startswith("@"):
            return part[1:]
        if part.startswith("UC") and len(part) >= 22:
            return part
    return parts[0] if parts else "channel"


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def flatten_entries(node: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not node:
        return []
    entries = node.get("entries")
    if entries:
        out: list[dict[str, Any]] = []
        for entry in entries:
            out.extend(flatten_entries(entry))
        return out
    video_id = node.get("id")
    if not video_id or not isinstance(video_id, str):
        return []
    if node.get("_type") == "playlist":
        return []
    if len(video_id) != 11:
        return []
    return [node]


def list_channel_videos(channel_url: str, max_videos: int | None) -> list[dict[str, str]]:
    url = channel_url
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    # Channel root returns Videos + Shorts. The /videos tab alone often
    # stops around 100 items even when the channel has more uploads.
    if parts and parts[0].startswith("@"):
        url = f"{parsed.scheme}://{parsed.netloc}/{parts[0]}"
        if parsed.query:
            url += f"?{parsed.query}"

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        "skip_download": True,
        "playlistend": max_videos,
    }
    LOG.info("Listing videos (yt-dlp): %s", url)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    videos: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in flatten_entries(info if isinstance(info, dict) else None):
        video_id = str(entry.get("id") or "")
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        title = str(entry.get("title") or video_id)
        videos.append(
            {
                "id": video_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        )
        if max_videos is not None and len(videos) >= max_videos:
            break
    return videos


def pick_transcript(transcript_list: Any, languages: list[str]) -> Any:
    try:
        return transcript_list.find_manually_created_transcript(languages)
    except NoTranscriptFound:
        pass
    try:
        return transcript_list.find_generated_transcript(languages)
    except NoTranscriptFound:
        pass
    try:
        return transcript_list.find_transcript(languages)
    except NoTranscriptFound:
        pass
    for transcript in transcript_list:
        return transcript
    raise NoTranscriptAvailable("no transcripts available")


def fetch_captions(
    api: YouTubeTranscriptApi,
    video_id: str,
    languages: list[str],
    retries: int,
    delay: float,
    cooldown: float,
) -> tuple[Any, Any]:
    last_error: Exception | None = None
    block_attempts = 8
    blocks = 0
    attempt = 0
    while True:
        try:
            transcript_list = api.list(video_id)
            transcript = pick_transcript(transcript_list, languages)
            fetched = transcript.fetch()
            return transcript, fetched
        except (IpBlocked, RequestBlocked) as exc:
            last_error = exc
            blocks += 1
            LOG.warning(
                "%s: YouTube IP block (%s/%s) — waiting %.0fs",
                video_id,
                blocks,
                block_attempts,
                cooldown,
            )
            if blocks >= block_attempts:
                raise IpStillBlocked(video_id) from exc
            time.sleep(cooldown)
        except RETRYABLE as exc:
            last_error = exc
            attempt += 1
            wait = delay * (2 ** (attempt - 1))
            LOG.warning(
                "%s: retryable error (%s/%s): %s — waiting %.1fs",
                video_id,
                attempt,
                retries + 1,
                exc.__class__.__name__,
                wait,
            )
            if attempt > retries:
                raise
            time.sleep(wait)
        except CouldNotRetrieveTranscript:
            raise
    assert last_error is not None
    raise last_error


def snippets_from_fetched(fetched: Any) -> list[Any]:
    if hasattr(fetched, "snippets"):
        return list(fetched.snippets)
    return list(fetched)


def snippet_text(snippet: Any) -> str:
    text = getattr(snippet, "text", None)
    if text is None and isinstance(snippet, dict):
        text = snippet.get("text", "")
    return re.sub(r"\s+", " ", str(text or "")).strip()


def snippet_start(snippet: Any) -> float:
    start = getattr(snippet, "start", None)
    if start is None and isinstance(snippet, dict):
        start = snippet.get("start", 0)
    try:
        return float(start or 0)
    except (TypeError, ValueError):
        return 0.0


def plain_text(fetched: Any) -> str:
    parts = [snippet_text(s) for s in snippets_from_fetched(fetched)]
    return " ".join(p for p in parts if p)


def timestamped_text(fetched: Any) -> str:
    lines: list[str] = []
    for snippet in snippets_from_fetched(fetched):
        text = snippet_text(snippet)
        if not text:
            continue
        lines.append(f"{format_timestamp(snippet_start(snippet))} {text}")
    return "\n".join(lines)


def caption_meta(transcript: Any, fetched: Any) -> tuple[str, str]:
    language = (
        getattr(fetched, "language_code", None)
        or getattr(transcript, "language_code", None)
        or getattr(fetched, "language", None)
        or getattr(transcript, "language", None)
        or "unknown"
    )
    is_generated = getattr(fetched, "is_generated", None)
    if is_generated is None:
        is_generated = getattr(transcript, "is_generated", None)
    caption_type = "auto-generated" if is_generated else "manual"
    return str(language), caption_type


def file_header(video: dict[str, str], language: str, caption_type: str) -> str:
    return (
        f"Title: {video['title']}\n"
        f"URL: {video['url']}\n"
        f"Video ID: {video['id']}\n"
        f"Language: {language}\n"
        f"Caption type: {caption_type}\n"
        "\n---\n\n"
    )


def skip_reason(exc: Exception) -> str:
    if isinstance(exc, TranscriptsDisabled):
        return "transcripts disabled / no Show transcript"
    if isinstance(exc, (NoTranscriptFound, NoTranscriptAvailable)):
        return "no matching transcript track"
    if isinstance(exc, (VideoUnavailable, InvalidVideoId)):
        return "video unavailable"
    if isinstance(exc, AgeRestricted):
        return "age restricted"
    if isinstance(exc, PoTokenRequired):
        return "YouTube required extra token (blocked)"
    if isinstance(exc, (IpBlocked, RequestBlocked)):
        return "IP / request blocked by YouTube"
    return str(exc) or exc.__class__.__name__


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "channel_url": "",
            "videos": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"channel_url": "", "videos": []}
    if not isinstance(data, dict):
        return {"channel_url": "", "videos": []}
    data.setdefault("videos", [])
    return data


def upsert_index(index: dict[str, Any], record: dict[str, Any]) -> None:
    videos = index.setdefault("videos", [])
    for i, existing in enumerate(videos):
        if existing.get("id") == record["id"]:
            videos[i] = record
            return
    videos.append(record)


def write_index(path: Path, index: dict[str, Any]) -> None:
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def already_exported(out_dir: Path, video_id: str) -> bool:
    return (out_dir / f"{video_id}.txt").exists() and (
        out_dir / f"{video_id}.timestamps.txt"
    ).exists()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    languages = [part.strip() for part in args.languages.split(",") if part.strip()]
    if not languages:
        languages = list(PREFERRED_LANGS)

    slug = channel_slug(args.channel_url)
    out_dir = args.out or Path("transcripts") / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.json"
    skipped_path = out_dir / "_skipped.txt"
    index = load_index(index_path)
    index["channel_url"] = args.channel_url

    videos = list_channel_videos(args.channel_url, args.max_videos)
    if not videos:
        LOG.error("No videos found. Try adding /videos to the channel URL.")
        return 1

    LOG.info("Found %d video(s). Saving transcripts to %s", len(videos), out_dir.resolve())
    try:
        proxy_config = build_proxy_config(args)
    except SystemExit as exc:
        LOG.error("%s", exc)
        return 1
    if proxy_config is None:
        LOG.info("No proxy configured; caption requests will use this machine's IP")
    api = YouTubeTranscriptApi(proxy_config=proxy_config)
    skipped: list[str] = []
    saved = 0
    resumed = 0

    for i, video in enumerate(videos, start=1):
        video_id = video["id"]
        prefix = f"[{i}/{len(videos)}] {video_id}"
        if already_exported(out_dir, video_id):
            LOG.info("%s already exported — skipping (resume)", prefix)
            resumed += 1
            existing = next(
                (row for row in index.get("videos", []) if row.get("id") == video_id),
                None,
            )
            if existing is None:
                upsert_index(
                    index,
                    {
                        "id": video_id,
                        "title": video["title"],
                        "url": video["url"],
                        "status": "ok",
                        "language": None,
                        "caption_type": None,
                        "plain_path": str(out_dir / f"{video_id}.txt"),
                        "timestamped_path": str(out_dir / f"{video_id}.timestamps.txt"),
                    },
                )
                write_index(index_path, index)
            continue

        try:
            transcript, fetched = fetch_captions(
                api,
                video_id,
                languages,
                args.retries,
                args.delay,
                args.cooldown,
            )
        except IpStillBlocked:
            LOG.error(
                "%s: YouTube is still blocking this IP. Stopping so already-saved "
                "files are kept; re-run the same command to resume.",
                prefix,
            )
            index["summary"] = {
                "listed": len(videos),
                "saved": saved,
                "resumed": resumed,
                "skipped": len(skipped),
                "stopped_on_ip_block": True,
                "stopped_at": video_id,
            }
            write_index(index_path, index)
            return 2
        except (CouldNotRetrieveTranscript, NoTranscriptAvailable) as exc:
            reason = skip_reason(exc)
            LOG.warning("%s skipped: %s", prefix, reason)
            skipped.append(f"{video_id}\t{video['title']}\t{reason}")
            upsert_index(
                index,
                {
                    "id": video_id,
                    "title": video["title"],
                    "url": video["url"],
                    "status": "skipped",
                    "reason": reason,
                    "language": None,
                    "caption_type": None,
                    "plain_path": None,
                    "timestamped_path": None,
                },
            )
            write_index(index_path, index)
            time.sleep(args.delay)
            continue
        except Exception as exc:  # last-resort: keep going through the playlist
            reason = skip_reason(exc)
            LOG.warning("%s skipped: %s", prefix, reason)
            skipped.append(f"{video_id}\t{video['title']}\t{reason}")
            upsert_index(
                index,
                {
                    "id": video_id,
                    "title": video["title"],
                    "url": video["url"],
                    "status": "skipped",
                    "reason": reason,
                    "language": None,
                    "caption_type": None,
                    "plain_path": None,
                    "timestamped_path": None,
                },
            )
            write_index(index_path, index)
            time.sleep(args.delay)
            continue

        language, caption_type = caption_meta(transcript, fetched)
        header = file_header(video, language, caption_type)
        plain_path = out_dir / f"{video_id}.txt"
        ts_path = out_dir / f"{video_id}.timestamps.txt"
        plain_path.write_text(header + plain_text(fetched) + "\n", encoding="utf-8")
        ts_path.write_text(header + timestamped_text(fetched) + "\n", encoding="utf-8")
        upsert_index(
            index,
            {
                "id": video_id,
                "title": video["title"],
                "url": video["url"],
                "status": "ok",
                "language": language,
                "caption_type": caption_type,
                "plain_path": str(plain_path),
                "timestamped_path": str(ts_path),
            },
        )
        write_index(index_path, index)
        saved += 1
        LOG.info("%s saved (%s, %s) %s", prefix, language, caption_type, video["title"][:80])
        time.sleep(args.delay)
        if args.batch_size > 0 and saved % args.batch_size == 0:
            LOG.info("Batch pause %.0fs after %s new saves", args.batch_pause, saved)
            time.sleep(args.batch_pause)

    if skipped:
        existing_skipped = ""
        if skipped_path.exists():
            existing_skipped = skipped_path.read_text(encoding="utf-8")
        new_lines = []
        for line in skipped:
            if line not in existing_skipped:
                new_lines.append(line)
        if new_lines:
            with skipped_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(new_lines) + "\n")

    index["summary"] = {
        "listed": len(videos),
        "saved": saved,
        "resumed": resumed,
        "skipped": len(skipped),
    }
    write_index(index_path, index)

    LOG.info(
        "Done. listed=%s saved=%s resumed=%s skipped=%s output=%s",
        len(videos),
        saved,
        resumed,
        len(skipped),
        out_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

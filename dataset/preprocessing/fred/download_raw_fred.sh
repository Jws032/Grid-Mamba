#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=$(dirname -- "$PROJECT_ROOT")

REPO_ID=${FRED_REPO_ID:-GabrieleMagrini/FRED}
REVISION=${FRED_REVISION:-main}
TARGET_DIR=${FRED_TARGET_DIR:-${WORKSPACE_ROOT}/datasets/FRED}
PROXY=${FRED_PROXY:-}
CONCURRENCY=${FRED_CONCURRENCY:-6}

STATE_DIR=${TARGET_DIR}/.download-state
MANIFEST=${STATE_DIR}/manifest.tsv
MANIFEST_JSON=${STATE_DIR}/tree.json
LOG_FILE=${FRED_LOG_FILE:-${STATE_DIR}/fred-download.log}
API_URL="https://huggingface.co/api/datasets/${REPO_ID}/tree/${REVISION}?recursive=true&expand=true&limit=100"
RESOLVE_PREFIX="https://huggingface.co/datasets/${REPO_ID}/resolve/${REVISION}/"

mkdir -p "$TARGET_DIR" "$STATE_DIR/done" "$STATE_DIR/active" "$STATE_DIR/file-logs"
touch "$LOG_FILE"
exec >> "$LOG_FILE" 2>&1

log() {
  printf '===== %s %s =====\n' "$(date -Is)" "$*"
}

proxy_args() {
  if [ -n "${PROXY:-}" ]; then
    printf '%s\n' --proxy "$PROXY"
  fi
}

auth_args() {
  if [ -n "${HF_TOKEN:-}" ]; then
    printf '%s\n' --header "Authorization: Bearer ${HF_TOKEN}"
  fi
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

generate_manifest() {
  log "fetch manifest from ${API_URL}"

  python3 - "$MANIFEST_JSON" "$MANIFEST" "$REPO_ID" "$REVISION" "$PROXY" <<'PY'
import json
import os
import re
import sys
import time
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

src, dst, repo_id, revision, proxy = sys.argv[1:6]
prefix = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
api = f"https://huggingface.co/api/datasets/{repo_id}/tree/{quote(revision, safe='')}?recursive=true&expand=true&limit=100"

handlers = []
if proxy:
    handlers.append(ProxyHandler({"http": proxy, "https": proxy}))
else:
    handlers.append(ProxyHandler({}))
opener = build_opener(*handlers)

headers = {"User-Agent": "fred-curl-downloader/1.0"}
token = os.environ.get("HF_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"

data = []
url = api
page = 0
while url:
    page += 1
    last_error = None
    for attempt in range(1, 11):
        try:
            req = Request(url, headers=headers)
            with opener.open(req, timeout=60) as resp:
                chunk = json.loads(resp.read().decode("utf-8"))
                link = resp.headers.get("Link", "")
            if isinstance(chunk, dict) and "error" in chunk:
                raise RuntimeError(chunk["error"])
            data.extend(chunk)
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
            print(f"manifest page={page} items={len(chunk)} total={len(data)}", flush=True)
            break
        except Exception as exc:
            last_error = exc
            if attempt == 10:
                raise
            time.sleep(min(60, attempt * 5))
    else:
        raise RuntimeError(last_error)

with open(src, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

rows = []
for item in data:
    if item.get("type") != "file":
        continue
    path = item.get("path", "")
    if path not in {".gitattributes", "README.md"} and not path.endswith(".zip"):
        continue
    size = item.get("size")
    size_text = "" if size is None else str(size)
    url = prefix + quote(path, safe="/")
    rows.append((path, size_text, url))

def rank(row):
    path = row[0]
    if path in {".gitattributes", "README.md"}:
        return (0, path)
    if path.startswith("test/"):
        return (1, path)
    return (2, path)

rows.sort(key=rank)

with open(dst, "w", encoding="utf-8") as f:
    for path, size_text, url in rows:
        f.write(f"{size_text}\t{path}\t{url}\n")
PY

  local count
  count=$(wc -l < "$MANIFEST")
  log "manifest ready count=${count} path=${MANIFEST}"
}

resolve_size() {
  local url=$1

  mapfile -t common_proxy_args < <(proxy_args)
  mapfile -t common_auth_args < <(auth_args)

  curl \
    --silent \
    --show-error \
    --head \
    --location \
    --fail \
    "${common_proxy_args[@]}" \
    "${common_auth_args[@]}" \
    --connect-timeout 30 \
    --retry 5 \
    --retry-delay 5 \
    --retry-all-errors \
    "$url" \
    | awk 'BEGIN { IGNORECASE=1 }
      /^x-linked-size:/ { gsub("\r", "", $2); x=$2 }
      /^content-length:/ { gsub("\r", "", $2); c=$2 }
      END {
        if (x ~ /^[0-9]+$/) print x;
        else if (c ~ /^[0-9]+$/) print c;
      }'
}

download_one() {
  local line=$1
  local expected rel url out done_marker active_marker file_log local_size status resolved

  IFS=$'\t' read -r expected rel url <<< "$line"
  [ -n "${rel:-}" ] || return 0
  [ -n "${url:-}" ] || return 0

  out="${TARGET_DIR}/${rel}"
  done_marker="${STATE_DIR}/done/${rel}.done"
  active_marker="${STATE_DIR}/active/${rel}.active"
  file_log="${STATE_DIR}/file-logs/${rel//\//__}.log"

  mkdir -p "$(dirname "$out")" "$(dirname "$done_marker")" "$(dirname "$active_marker")"

  if [[ "$rel" == *.zip ]] && ! is_positive_int "$expected"; then
    resolved=$(resolve_size "$url" || true)
    if is_positive_int "$resolved"; then
      expected=$resolved
    fi
  fi

  local_size=0
  if [ -f "$out" ]; then
    local_size=$(wc -c < "$out")
  fi

  if is_positive_int "$expected" && [ "$local_size" -eq "$expected" ]; then
    printf '%s size=%s\n' "$(date -Is)" "$local_size" > "$done_marker"
    rm -f "$active_marker"
    log "skip complete ${rel} size=${local_size}"
    return 0
  fi

  mapfile -t common_proxy_args < <(proxy_args)
  mapfile -t common_auth_args < <(auth_args)

  while true; do
    printf '%s pid=%s rel=%s local=%s expected=%s\n' \
      "$(date -Is)" "$$" "$rel" "$local_size" "${expected:-unknown}" > "$active_marker"
    log "start ${rel} local=${local_size} expected=${expected:-unknown}"

    set +e
    curl \
      --silent \
      --show-error \
      --location \
      --fail \
      "${common_proxy_args[@]}" \
      "${common_auth_args[@]}" \
      --connect-timeout 30 \
      --speed-time 180 \
      --speed-limit 1024 \
      --retry 20 \
      --retry-delay 15 \
      --retry-all-errors \
      --continue-at - \
      --output "$out" \
      --write-out "curl_status=%{exitcode} http_code=%{http_code} size_download=%{size_download} speed_download=%{speed_download} time_total=%{time_total}\n" \
      "$url" >> "$file_log" 2>&1

    status=$?
    set -e
    local_size=0
    if [ -f "$out" ]; then
      local_size=$(wc -c < "$out")
    fi

    if [ "$status" -eq 0 ]; then
      if is_positive_int "$expected" && [ "$local_size" -ne "$expected" ]; then
        log "size mismatch ${rel} local=${local_size} expected=${expected}; retry in 60s"
        sleep 60
        continue
      fi

      printf '%s size=%s\n' "$(date -Is)" "$local_size" > "$done_marker"
      rm -f "$active_marker"
      log "done ${rel} size=${local_size}"
      return 0
    fi

    if ! is_positive_int "$expected"; then
      resolved=$(resolve_size "$url" || true)
      if is_positive_int "$resolved"; then
        expected=$resolved
      fi
    fi

    if [ "$status" -eq 22 ] && is_positive_int "$expected" && [ "$local_size" -eq "$expected" ]; then
      printf '%s size=%s\n' "$(date -Is)" "$local_size" > "$done_marker"
      rm -f "$active_marker"
      log "done ${rel} size=${local_size} after http 416"
      return 0
    fi

    log "retry ${rel} status=${status} local=${local_size} expected=${expected:-unknown} sleep=60s"
    sleep 60
  done
}

export REPO_ID REVISION TARGET_DIR PROXY CONCURRENCY STATE_DIR MANIFEST MANIFEST_JSON LOG_FILE API_URL RESOLVE_PREFIX
export -f log proxy_args auth_args is_positive_int generate_manifest resolve_size download_one

log "FRED download start"
log "target=${TARGET_DIR}"
log "proxy=${PROXY:-none}"
log "concurrency=${CONCURRENCY}"

if [ ! -s "$MANIFEST" ] || [ "${FRED_REFRESH_MANIFEST:-0}" = "1" ]; then
  generate_manifest
fi

if [ "${FRED_ONLY_MANIFEST:-0}" = "1" ]; then
  log "manifest-only mode complete"
  exit 0
fi

xargs -r -d '\n' -n 1 -P "$CONCURRENCY" bash -c 'download_one "$1"' _ < "$MANIFEST"

log "FRED download complete"

#!/usr/bin/env bash
# Cloudflare Pages build script for the Flet web bundle.
#
# Pages' default Linux build image has Python + pip but NOT Flutter.
# We download a pinned Flutter SDK into the build sandbox each run, then
# use `flet build web` to emit a static bundle into `build/web/`.
#
# Cloudflare Pages → Build settings:
#   Build command:    bash cloudflare-build.sh
#   Output directory: build/web
#   Root directory:   (blank — repo root)

set -euo pipefail

FLUTTER_VERSION="3.27.4"
FLUTTER_TGZ="flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/${FLUTTER_TGZ}"
FLUTTER_DIR="${HOME}/flutter"

echo "→ Cloudflare Pages build for FinNavigator (Flet web)"
echo "  Python: $(python --version 2>&1 || true)"
echo "  Pip:    $(pip --version 2>&1 || true)"

# 1. Install Flutter (cached at $HOME/flutter between builds when Pages keeps the cache warm)
if [ ! -x "${FLUTTER_DIR}/bin/flutter" ]; then
    echo "→ Downloading Flutter ${FLUTTER_VERSION}…"
    curl -fsSL "${FLUTTER_URL}" -o "/tmp/${FLUTTER_TGZ}"
    mkdir -p "${HOME}"
    tar -xJf "/tmp/${FLUTTER_TGZ}" -C "${HOME}"
    rm "/tmp/${FLUTTER_TGZ}"
fi

export PATH="${FLUTTER_DIR}/bin:${PATH}"
flutter --version
flutter config --no-analytics --enable-web

# 2. Install Python deps for the UI build
pip install --upgrade pip
pip install -r requirements-ui.txt

# 3. Build the Flet web bundle
flet --version
flet build web \
    --project finnavigator \
    --product "FinNavigator" \
    --org "io.moh749.finnav" \
    --description "FinNavigator — multi-agent financial intelligence" \
    --no-rich-output

ls -la build/web/ | head -20
echo "→ Build complete. Cloudflare Pages should serve build/web/"

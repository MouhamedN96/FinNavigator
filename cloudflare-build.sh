#!/usr/bin/env bash
# Cloudflare Pages build script for the Flet web bundle.
#
# Pages settings:
#   Build command:    bash cloudflare-build.sh
#   Output directory: build/web
#   Root directory:   (blank)
#   Environment vars (recommended):
#     PYTHON_VERSION = 3.11
#     FLUTTER_VERSION = 3.27.4
#
# Diagnostic-friendly: every step prints a banner so the failed step is
# obvious in Cloudflare's build log.

set -eu
set -o pipefail

banner() { printf "\n========== %s ==========\n" "$1"; }

banner "ENV"
echo "PWD       : $(pwd)"
echo "USER      : $(whoami 2>/dev/null || echo unknown)"
echo "HOME      : ${HOME:-unset}"
echo "PYTHON    : $(command -v python || command -v python3 || echo MISSING)"
$(command -v python || command -v python3) --version || true
echo "PIP       : $(command -v pip || command -v pip3 || echo MISSING)"
echo "BASH      : ${BASH_VERSION:-not bash}"
echo "DISK      : $(df -h . | tail -1)"

PY=$(command -v python3 || command -v python)
PIP=$(command -v pip3 || command -v pip)

FLUTTER_VERSION="${FLUTTER_VERSION:-3.27.4}"
FLUTTER_TGZ="flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/${FLUTTER_TGZ}"
FLUTTER_DIR="${HOME}/flutter"

banner "STEP 1 — Flutter ${FLUTTER_VERSION}"
if [ -x "${FLUTTER_DIR}/bin/flutter" ]; then
    echo "Using cached Flutter at ${FLUTTER_DIR}"
else
    echo "Downloading from ${FLUTTER_URL}"
    if ! curl -fSL "${FLUTTER_URL}" -o "/tmp/${FLUTTER_TGZ}"; then
        echo "✗ Flutter download failed. URL may have moved."
        echo "  Try a different FLUTTER_VERSION (e.g. 3.24.5, 3.32.0) in Pages env vars."
        exit 1
    fi
    mkdir -p "${HOME}"
    tar -xJf "/tmp/${FLUTTER_TGZ}" -C "${HOME}"
    rm "/tmp/${FLUTTER_TGZ}"
fi
export PATH="${FLUTTER_DIR}/bin:${PATH}"
flutter --version
flutter config --no-analytics --enable-web

banner "STEP 2 — pip install requirements-ui.txt"
"${PIP}" install --upgrade pip
"${PIP}" install -r requirements-ui.txt
"${PY}" -c "import flet; print('flet', flet.__version__)"

# Cloudflare Pages uses asdf-managed Python and doesn't put its scripts
# dir on PATH. Find where pip dropped the `flet` entry point and prepend it.
PY_BIN_DIR="$(dirname "${PY}")"
PY_SCRIPTS_DIR="$("${PY}" -c "import sysconfig; print(sysconfig.get_path('scripts'))")"
export PATH="${PY_SCRIPTS_DIR}:${PY_BIN_DIR}:${PATH}"
echo "PY_BIN_DIR     : ${PY_BIN_DIR}"
echo "PY_SCRIPTS_DIR : ${PY_SCRIPTS_DIR}"
echo "which flet     : $(command -v flet || echo NOT-ON-PATH)"

# Resolve flet binary explicitly with shutil fallback so we don't trust PATH alone
FLET_BIN="$("${PY}" -c "import shutil; print(shutil.which('flet') or '')")"
if [ -z "${FLET_BIN}" ]; then
    for candidate in "${PY_SCRIPTS_DIR}/flet" "${PY_BIN_DIR}/flet"; do
        if [ -x "${candidate}" ]; then FLET_BIN="${candidate}"; break; fi
    done
fi
[ -z "${FLET_BIN}" ] && { echo "✗ flet binary not found anywhere"; exit 1; }
echo "FLET_BIN       : ${FLET_BIN}"

banner "STEP 3 — flet build web"
"${FLET_BIN}" build web ui/app.py \
    --project finnavigator \
    --product "FinNavigator" \
    --org "io.moh749.finnav" \
    --description "FinNavigator — multi-agent financial intelligence" \
    --no-rich-output

banner "OUTPUT"
ls -la build/web/ | head -30
du -sh build/web 2>/dev/null || true
echo "✓ Build complete"

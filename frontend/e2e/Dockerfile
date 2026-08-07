# Headless-browser runner for the UI check.
#
# Built rather than pulled: ghcr.io/puppeteer/puppeteer ships amd64 only, and its bundled
# Chrome cannot start under QEMU emulation on an arm64 host (it times out waiting for the
# DevTools WS endpoint). Debian's chromium package is native on both architectures, so we
# use puppeteer-core and point it at the system binary instead of downloading one.
FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        fonts-liberation \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_SKIP_DOWNLOAD=1 \
    CHROME_PATH=/usr/bin/chromium

# The package ends up at /node_modules. ESM ignores NODE_PATH entirely; it resolves bare
# specifiers by walking up parent directories looking for node_modules, so a script
# bind-mounted at /work/ui-check.mjs finds /node_modules but never a global install under
# /usr/local/lib. Installed in a scratch directory and moved, because `npm install
# --prefix /` collides with the root-owned npm tree ("Tracker idealTree already exists").
RUN mkdir -p /tmp/inst && cd /tmp/inst \
    && npm install puppeteer-core@23.11.1 \
    && mv /tmp/inst/node_modules /node_modules \
    && rm -rf /tmp/inst /root/.npm

WORKDIR /work
CMD ["node", "ui-check.mjs"]

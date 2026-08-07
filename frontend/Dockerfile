# syntax=docker/dockerfile:1.7
#
# Frontend image: build the Vite bundle, then serve it from nginx.
# nginx also reverse-proxies /api to the backend, so the browser only ever talks to one
# origin — no CORS in production, and the same relative URLs work in dev.

# --------------------------------------------------------------------------------------
# Stage 1: build
# --------------------------------------------------------------------------------------
FROM node:22-alpine AS builder

WORKDIR /app

# Copy manifests first so `npm ci` is cached independently of source changes.
COPY package.json package-lock.json* ./

# `npm ci` needs a lockfile; fall back to `install` on first build when none is committed.
RUN --mount=type=cache,target=/root/.npm \
    if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

# --------------------------------------------------------------------------------------
# Stage 2: serve
# --------------------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# Explicit 127.0.0.1:80 — busybox wget resolves "localhost" to ::1 first, and nginx here
# listens on IPv4 only, so the bare hostname form fails with "connection refused".
HEALTHCHECK --interval=30s --timeout=4s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:80/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]

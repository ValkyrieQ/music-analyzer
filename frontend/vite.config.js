import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, proxy /api to the backend container so the browser sees a single origin
// (no CORS, and relative URLs work identically in dev and in the nginx-served build).
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://api:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})

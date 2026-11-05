import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The built viewer ships inside the Python package, so `pip install jepa-scope` never needs npm.
// During frontend work, run `jepa-scope view x.latent --no-browser` and `npm run dev` side by side.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/jepa_scope/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/video': 'http://127.0.0.1:8765',
    },
  },
})

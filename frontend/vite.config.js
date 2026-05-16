import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // Proxy API calls to Flask in dev so CORS is never an issue.
  // Any request to /health, /predict, /model-info etc. is
  // forwarded to Flask on port 5000.
  server: {
    port: 3000,
    proxy: {
      '/health':            'http://localhost:5000',
      '/model-info':        'http://localhost:5000',
      '/predict':           'http://localhost:5000',
      '/predict-with-mask': 'http://localhost:5000',
    },
  },
})
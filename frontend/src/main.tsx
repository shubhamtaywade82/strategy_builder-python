import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router'
import './index.css'
import { AppQueryProvider } from "@/providers/query"
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <BrowserRouter>
    <AppQueryProvider>
      <App />
    </AppQueryProvider>
  </BrowserRouter>,
)

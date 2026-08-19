import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import './App.css'
import { ChatShell } from './components/ChatShell'
import { RunInspector } from './components/RunInspector'
import { SettingsPanel } from './components/SettingsPanel'
import { Sidebar } from './components/Sidebar'

const queryClient = new QueryClient()

function App() {
  const [isSettingsOpen, setSettingsOpen] = useState(false)

  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex min-h-screen bg-slate-100 text-slate-950">
        <Sidebar onOpenSettings={() => setSettingsOpen(true)} />
        <ChatShell />
        <RunInspector />
        <SettingsPanel open={isSettingsOpen} onClose={() => setSettingsOpen(false)} />
      </div>
    </QueryClientProvider>
  )
}

export default App

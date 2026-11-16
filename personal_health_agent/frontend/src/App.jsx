import React, { useState, useEffect } from 'react'
import SplashScreen from './components/SplashScreen'
import ChatInterface from './components/ChatInterface'
import './App.css'

// API 使用相对路径，开发时由 Vite 代理。
const API_BASE = '/api'
const ACTIVE_SESSION_KEY = 'pha_active_session'

function App() {
  const [config, setConfig] = useState(null)
  const [session, setSession] = useState(null)
  const [userId, setUserId] = useState('test')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // 挂载时获取配置
  useEffect(() => {
    fetchConfig()
  }, [])

  const clearActiveSession = () => {
    localStorage.removeItem(ACTIVE_SESSION_KEY)
  }

  const saveActiveSession = (sessionData, activeUserId) => {
    const payload = {
      session_id: sessionData.session_id,
      user_id: activeUserId || sessionData?.config?.user_id || 'test',
    }
    localStorage.setItem(ACTIVE_SESSION_KEY, JSON.stringify(payload))
  }

  const restoreSession = async (latestConfig) => {
    const raw = localStorage.getItem(ACTIVE_SESSION_KEY)
    if (!raw) return false

    let storedSession
    try {
      storedSession = JSON.parse(raw)
    } catch {
      clearActiveSession()
      return false
    }

    if (!storedSession?.session_id) {
      clearActiveSession()
      return false
    }

    try {
      const res = await fetch(`${API_BASE}/sessions/${storedSession.session_id}`)
      if (!res.ok) throw new Error('已保存的会话不可用')
      const restoredState = await res.json()

      // 模型从目录下架后，结束保存的会话并返回配置页。
      const restoredProvider = restoredState.config?.provider
      const restoredModel = restoredState.config?.model_id
      const cfg = latestConfig || config
      const providerModels = (restoredProvider && cfg?.models?.[restoredProvider]) || []
      const restoredModelExists = providerModels.some(m => m.id === restoredModel)
      if (!restoredModelExists) {
        await fetch(`${API_BASE}/sessions/${storedSession.session_id}`, { method: 'DELETE' }).catch(() => {})
        clearActiveSession()
        return false
      }

      setSession({
        session_id: restoredState.session_id,
        config: restoredState.config,
        created_at: restoredState.created_at,
        status: restoredState.status,
      })
      setUserId(restoredState.config?.user_id || storedSession.user_id || 'test')
      return true
    } catch {
      clearActiveSession()
      return false
    }
  }

  const fetchConfig = async () => {
    try {
      setLoading(true)
      const res = await fetch(`${API_BASE}/config`)
      if (!res.ok) throw new Error('无法获取门户配置')
      const data = await res.json()
      setConfig(data)
      await restoreSession(data)
      setLoading(false)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const createSession = async (sessionConfig) => {
    try {
      setLoading(true)
      // 从会话配置保存用户 ID。
      if (sessionConfig.user_id) {
        setUserId(sessionConfig.user_id)
      }
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionConfig),
      })
      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || '无法创建会话')
      }
      const data = await res.json()
      setSession(data)
      saveActiveSession(data, sessionConfig.user_id)
      setLoading(false)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const endSession = async () => {
    if (session) {
      try {
        await fetch(`${API_BASE}/sessions/${session.session_id}`, {
          method: 'DELETE',
        })
      } catch (err) {
        console.error('无法结束会话：', err)
      }
    }
    setSession(null)
    clearActiveSession()
  }

  const handleRefresh = async () => {
    if (!session) return

    const sessionConfig = session.config
    setError(null)
    setLoading(true)

    try {
      try {
        await fetch(`${API_BASE}/sessions/${session.session_id}`, {
          method: 'DELETE',
        })
      } catch (err) {
        console.error('重新开始时无法结束旧会话：', err)
      }

      clearActiveSession()

      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionConfig),
      })
      if (!res.ok) {
        let detail = '无法重新开始会话'
        try {
          const errData = await res.json()
          detail = errData.detail || detail
        } catch {
          // 如果响应不是 JSON，使用默认错误提示。
        }
        throw new Error(detail)
      }

      const data = await res.json()
      setSession(data)
      setUserId(sessionConfig.user_id || data?.config?.user_id || 'test')
      saveActiveSession(data, sessionConfig.user_id)
    } catch (err) {
      setSession(null)
      setError(err.message)
      clearActiveSession()
    } finally {
      setLoading(false)
    }
  }

  // 加载状态
  if (loading && !config) {
    return (
      <div className="app-loading">
        <div className="spinner"></div>
        <p>正在加载个人健康 Agent…</p>
      </div>
    )
  }

  // 错误状态
  if (error && !config) {
    return (
      <div className="app-error">
        <h2>⚠️ 连接失败</h2>
        <p>{error}</p>
        <p className="hint">请确认 API 服务已启动。</p>
        <code>cd personal_health_agent && uvicorn api.main:app --reload</code>
        <button onClick={() => { setError(null); fetchConfig(); }}>
          重试
        </button>
      </div>
    )
  }

  return (
    <div className="app">
      {!session ? (
        <SplashScreen
          config={config}
          onStart={createSession}
          loading={loading}
          error={error}
        />
      ) : (
        <ChatInterface
          session={session}
          personaName={config?.personas?.find(p => p.id === session.config.persona_id)?.name}
          apiBase={API_BASE}
          onEndSession={endSession}
          onRefresh={handleRefresh}
          refreshing={loading}
          userId={userId}
          userStudyMode={config?.user_study_mode || false}
        />
      )}
    </div>
  )
}

export default App

import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatInterface.css'


const TAG_LABELS = {
  Orch:     { label: '编排器',   icon: '🎯' },
  DS:       { label: '数据科学',   icon: '📊' },
  Coach:    { label: '健康教练',   icon: '💬' },
  DE:       { label: '领域专家',  icon: '🩺' },
  LLM:      { label: '模型调用',       icon: '🧠' },
  Parallel: { label: '并行架构',       icon: '⚡' },
  PHIA:     { label: 'PHIA',     icon: '🤖' },
}

// 处理时显示当前步骤，处理完成后可展开全部步骤。
function ThinkingIndicator({ steps, isActive, elapsed }) {
  const [expanded, setExpanded] = useState(false)

  // 实时摘要只展示业务步骤，展开后仍能看到底层模型调用。
  const displaySteps = steps.filter(s => s.tag !== 'LLM')
  const lastDisplayStep = displaySteps[displaySteps.length - 1]

  if (steps.length === 0 && isActive) {
    return (
      <div className="thinking-indicator active">
        <span className="thinking-spinner" />
        <span className="thinking-label">正在处理…</span>
      </div>
    )
  }

  // 只有底层模型调用时，显示通用加载提示。
  if (!lastDisplayStep && isActive) {
    return (
      <div className="thinking-indicator active">
        <span className="thinking-spinner" />
        <span className="thinking-label">正在处理…</span>
      </div>
    )
  }

  if (!lastDisplayStep) return null

  const lastTag = TAG_LABELS[lastDisplayStep?.tag] || { label: lastDisplayStep?.tag, icon: '⚙️' }

  return (
    <div className={`thinking-indicator ${isActive ? 'active' : 'done'}`}>
      {/* 主行——完成后可点击扩展 */}
      <button className="thinking-row" onClick={() => !isActive && setExpanded(!expanded)}>
        {isActive ? (
          <>
            <span className="thinking-spinner" />
            <span className="thinking-label">
              {lastTag.icon} {lastDisplayStep.message}
            </span>
            <span className="thinking-elapsed">{elapsed} 秒</span>
          </>
        ) : (
          <>
            <span className="thinking-check">✓</span>
            <span className="thinking-label done-label">
              {lastTag.icon} {lastDisplayStep.message}
            </span>
            <span className="thinking-elapsed">{elapsed} 秒</span>
            {steps.length > 1 && (
              <span className={`thinking-chevron ${expanded ? 'open' : ''}`}>▾</span>
            )}
          </>
        )}
      </button>

      {/* 扩展步骤列表——显示ALL步骤，包括LLM调用 */}
      {expanded && !isActive && steps.length > 1 && (
        <div className="thinking-expanded">
          {steps.map((step, i) => {
            const tag = TAG_LABELS[step.tag] || { label: step.tag, icon: '⚙️' }
            return (
              <div key={i} className="thinking-exp-step">
                <span className="exp-icon">{tag.icon}</span>
                <span className="exp-msg">{step.message}</span>
                <span className="exp-time">{step.timestamp}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function ChatInterface({ session, personaName, apiBase, onEndSession, onRefresh, refreshing, userId, userStudyMode }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [thinkingSteps, setThinkingSteps] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const messagesEndRef = useRef(null)

  // 用于格式化对话日志中显示的时间戳的函数
  const formatTimestampForLog = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  // 将对话日志保存为txt文件的函数
  const saveConversationLog = async () => {
    // 格式化对话日志
    let logContent = ''
    messages.forEach(msg => {
      const role = msg.role === 'user' ? '**User**' : '**Agent**'
      const timestamp = formatTimestampForLog(msg.timestamp)
      logContent += `${role} [${timestamp}]: ${msg.content}\n\n`
    })

    // 创建文件名：[用户 id] - [代理名称] - [hh-mm] - [MM-DD-YYYY].txt
    const now = new Date()
    const hours = String(now.getHours()).padStart(2, '0')
    const minutes = String(now.getMinutes()).padStart(2, '0')
    const seconds = String(now.getSeconds()).padStart(2, '0')
    const month = String(now.getMonth() + 1).padStart(2, '0')
    const day = String(now.getDate()).padStart(2, '0')
    const year = now.getFullYear()
    
    const agentName = session.config.baseline
    const filename = `${userId}-${agentName}-${hours}-${minutes}-${seconds}-${month}-${day}-${year}.txt`

    // (raw_conversation.log + summary.json) 处理所有的持久性。
  }

  // 处理结束会话 - 保存日志然后调用父调用
  const handleEndSession = async () => {
    await saveConversationLog()
    onEndSession()
  }
  const timerRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinkingSteps])

  useEffect(() => {
    let isMounted = true

    const loadMessages = async () => {
      try {
        const res = await fetch(`${apiBase}/sessions/${session.session_id}/messages`)
        if (!res.ok) throw new Error('无法加载会话消息')
        const data = await res.json()

        if (!isMounted) return

        if (Array.isArray(data) && data.length > 0) {
          setMessages(data)
        } else {
          setMessages([{
            role: 'assistant',
            content: getWelcomeMessage(),
            timestamp: new Date().toISOString(),
          }])
        }
      } catch {
        if (!isMounted) return
        setMessages([{
          role: 'assistant',
          content: getWelcomeMessage(),
          timestamp: new Date().toISOString(),
        }])
      }
    }

    loadMessages()

    return () => {
      isMounted = false
    }
  }, [apiBase, session.session_id, userStudyMode])

  // 倾斜计时器
  useEffect(() => {
    if (loading) {
      setElapsed(0)
      timerRef.current = setInterval(() => {
        setElapsed(prev => prev + 1)
      }, 1000)
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [loading])

  const getWelcomeMessage = () => {
    // 在用户学习模式下使用匿名姓名( A/B/C 代理)
    const baselineNames = userStudyMode
      ? {
          pha: '系统 A',
          parallel: '系统 B',
          phia: '系统 C',
        }
      : {
          pha: 'PHA（多 Agent 编排）',
          parallel: 'Parallel（多 Agent 并行）',
          phia: 'PHIA（单 Agent）',
        }
    return `👋 你好，我是你的个人健康 Agent，当前使用 **${baselineNames[session.config.baseline]}**。你今天想了解什么？`
  }

  const sendMessage = useCallback(async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString(),
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setIsStreaming(true)
    setThinkingSteps([])
    setError(null)

    try {
      const res = await fetch(`${apiBase}/sessions/${session.session_id}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content }),
      })

      if (!res.ok) {
        let errDetail
        try {
          const errData = await res.json()
          errDetail = errData.detail || JSON.stringify(errData)
        } catch {
          errDetail = `HTTP ${res.status}: ${res.statusText}`
        }
        throw new Error(errDetail)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()

        for (const part of parts) {
          if (!part.trim()) continue

          const lines = part.split('\n')
          let eventType = 'step'
          let data = null

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              try { data = JSON.parse(line.slice(6)) } catch {}
            }
          }

          if (!data) continue

          if (eventType === 'step') {
            setThinkingSteps(prev => [...prev, data])
          } else if (eventType === 'result') {
            setIsStreaming(false)
            setThinkingSteps(currentSteps => {
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.content,
                timestamp: new Date().toISOString(),
                processingTime: data.processing_time_ms,
                thinkingSteps: currentSteps,
              }])
              return []
            })
          } else if (eventType === 'error') {
            setIsStreaming(false)
            const errorContent = `⚠️ **服务端出错**\n\n\`\`\`\n${data.detail}\n\`\`\`\n\n可在服务端日志中查看详细错误。`
            setThinkingSteps(currentSteps => {
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: errorContent,
                timestamp: new Date().toISOString(),
                isError: true,
                thinkingSteps: currentSteps,
              }])
              return []
            })
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err.message)
      setIsStreaming(false)
      const errorContent = `⚠️ **请求失败**\n\n\`\`\`\n${err.message}\n\`\`\``
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorContent,
        timestamp: new Date().toISOString(),
        isError: true,
      }])
      setThinkingSteps([])
    } finally {
      setLoading(false)
    }
  }, [input, loading, apiBase, session.session_id])

  const formatTime = (timestamp) => {
    if (!timestamp) return ''
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const renderMessage = (msg, index) => {
    return (
      <div
        key={index}
        className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}
      >
        <div className="message-avatar">
          {msg.role === 'user' ? '👤' : '🤖'}
        </div>
        <div className="message-content">
          {/* 已完成的思考摘要 */}
          {msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
            <ThinkingIndicator
              steps={msg.thinkingSteps}
              isActive={false}
              elapsed={msg.processingTime ? (msg.processingTime / 1000).toFixed(0) : '?'}
            />
          )}
          <div className="message-text markdown-body">
            {msg.role === 'user' ? (
              <p>{msg.content}</p>
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            )}
          </div>
          <div className="message-meta">
            <span className="message-time">{formatTime(msg.timestamp)}</span>
            {msg.processingTime && (
              <span className="message-processing">
                {(msg.processingTime / 1000).toFixed(1)} 秒
              </span>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="chat-header-info">
          <h1>🏥 个人健康对话</h1>
          <div className="session-info">
            <span className="session-badge baseline">
              {userStudyMode
                ? { pha: '系统 A', parallel: '系统 B', phia: '系统 C' }[session.config.baseline]
                : session.config.baseline.toUpperCase()}
            </span>
            {!userStudyMode && (
              <>
                <span className="session-badge model">
                  {session.config.model_id.split('/').pop()}
                </span>
                <span className="session-badge persona">
                  {personaName || session.config.persona_id}
                </span>
              </>
            )}
          </div>
        </div>
        <div className="chat-header-actions">
          <button className="refresh-btn" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? '正在重新开始…' : '重新开始'}
          </button>
          <button className="end-session-btn" onClick={handleEndSession} disabled={refreshing}>
            结束会话
          </button>
        </div>
      </header>

      <div className="pha-disclaimer" role="note">
        <span className="pha-disclaimer-label">提示：</span>
        本门户仅供研究使用，不作为医疗器械，也不适用于任何临床用途，包括诊断、预后判断和治疗建议。
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => renderMessage(msg, i))}

        {/* 展示服务端实时处理进度 */}
        {isStreaming && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <ThinkingIndicator
                steps={thinkingSteps}
                isActive={true}
                elapsed={elapsed}
              />
            </div>
          </div>
        )}

        {/* 等待服务端发出第一条事件时显示加载状态 */}
        {loading && !isStreaming && thinkingSteps.length === 0 && (
          <div className="message assistant loading">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={sendMessage}>
        <div className="chat-input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="想了解哪些健康数据？"
            disabled={loading}
            autoFocus
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? '发送中…' : '发送'}
          </button>
        </div>
        <p className="input-hint">
          按 Enter 发送 · 会话 ID： {session.session_id}
        </p>
      </form>
    </div>
  )
}

export default ChatInterface

import React, { useState, useEffect, useMemo, useRef } from 'react'
import './SplashScreen.css'

const API_BASE = '/api'

// localStorage 的存储键
const STORAGE_KEYS = {
  gemini: 'pha_gemini_key',
  openai: 'pha_openai_key',
  anthropic: 'pha_anthropic_key',
  tavily: 'pha_tavily_key',
  provider: 'pha_provider',
  baseline: 'pha_baseline',
  persona: 'pha_persona',
  model: 'pha_model',
  userId: 'pha_user_id',
}

function SplashScreen({ config, onStart, loading, error }) {
  const userStudyMode = config?.user_study_mode ?? false

  // 从后端获取的用户研究默认值( config/user_study_defaults.json)
  const [userStudyDefaults, setUserStudyDefaults] = useState(null)

  // 从 localStorage 装入保存的值
  const savedProvider = localStorage.getItem(STORAGE_KEYS.provider) || config?.defaults?.provider || 'gemini'
  const savedBaseline = localStorage.getItem(STORAGE_KEYS.baseline) || config?.defaults?.baseline || 'pha'
  const savedPersona = localStorage.getItem(STORAGE_KEYS.persona) || config?.defaults?.persona || 'sample'
  const savedModel = localStorage.getItem(STORAGE_KEYS.model) || config?.defaults?.model || ''
  const savedUserId = localStorage.getItem(STORAGE_KEYS.userId) || 'test'
  
  const [provider, setProvider] = useState(savedProvider)
  const [modelId, setModelId] = useState(savedModel)
  const [baseline, setBaseline] = useState(savedBaseline)
  const [persona, setPersona] = useState(savedPersona)
  const [userId, setUserId] = useState(savedUserId)
  const [apiKeys, setApiKeys] = useState({
    gemini: localStorage.getItem(STORAGE_KEYS.gemini) || '',
    openai: localStorage.getItem(STORAGE_KEYS.openai) || '',
    anthropic: localStorage.getItem(STORAGE_KEYS.anthropic) || '',
    tavily: localStorage.getItem(STORAGE_KEYS.tavily) || '',
  })
  const [validationError, setValidationError] = useState(null)
  const [dynamicModels, setDynamicModels] = useState(null)
  const [checkingAvailability, setCheckingAvailability] = useState(false)
  const isFirstAvailabilityRun = useRef(true)

  // 在用户研究模式下获取用户研究默认配置值
  useEffect(() => {
    if (!userStudyMode) return
    const fetchDefaults = async () => {
      try {
        const res = await fetch(`${API_BASE}/user-study-defaults`)
        if (res.ok) {
          const data = await res.json()
          setUserStudyDefaults(data)
          if (data.provider) setProvider(data.provider)
          if (data.model) setModelId(data.model)
        }
      } catch (err) {
        console.warn('Could not fetch user study defaults:', err)
      }
    }
    fetchDefaults()
  }, [userStudyMode])

  // 首次打开页面时立即校验已保存的密钥。之后等待输入停止 500 毫秒再请求。
  // 没有填写密钥时，继续显示默认模型列表。
  useEffect(() => {
    const hasAnyKey = !!(apiKeys.gemini || apiKeys.openai || apiKeys.anthropic)
    if (!hasAnyKey) {
      setDynamicModels(null)
      setCheckingAvailability(false)
      return
    }

    const fetchModels = async () => {
      setCheckingAvailability(true)
      try {
        const response = await fetch(`${API_BASE}/models`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            gemini_api_key: apiKeys.gemini || null,
            openai_api_key: apiKeys.openai || null,
            anthropic_api_key: apiKeys.anthropic || null,
          }),
        })
        if (response.ok) {
          const data = await response.json()
          setDynamicModels(data)
        }
      } catch (err) {
        console.log('Could not fetch dynamic models:', err)
      } finally {
        setCheckingAvailability(false)
      }
    }

    if (isFirstAvailabilityRun.current) {
      isFirstAvailabilityRun.current = false
      fetchModels()
      return
    }

    const timer = setTimeout(fetchModels, 500)
    return () => clearTimeout(timer)
  }, [userStudyMode, apiKeys.gemini, apiKeys.openai, apiKeys.anthropic])

  // 如果可用，使用动态模型，否则会回到静态配置
  const effectiveConfig = useMemo(() => ({
    ...config,
    models: dynamicModels || config?.models,
  }), [config, dynamicModels])

  // API当它们改变时，保存 localStorage 的密钥
  useEffect(() => {
    if (apiKeys.gemini) localStorage.setItem(STORAGE_KEYS.gemini, apiKeys.gemini)
    if (apiKeys.openai) localStorage.setItem(STORAGE_KEYS.openai, apiKeys.openai)
    if (apiKeys.anthropic) localStorage.setItem(STORAGE_KEYS.anthropic, apiKeys.anthropic)
    if (apiKeys.tavily) localStorage.setItem(STORAGE_KEYS.tavily, apiKeys.tavily)
  }, [apiKeys])

  // 将首选项保存到 localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.provider, provider)
  }, [provider])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.baseline, baseline)
  }, [baseline])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.persona, persona)
  }, [persona])

  // 保存 userId 改为： localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.userId, userId)
  }, [userId])

  // 将模型保存到 localStorage
  useEffect(() => {
    if (modelId) {
      localStorage.setItem(STORAGE_KEYS.model, modelId)
    }
  }, [modelId])

  // 普通模式优先恢复保存的模型，其次选择第一个可用模型，最后回退到列表首项。
  useEffect(() => {
    if (userStudyMode || !effectiveConfig?.models) return

    const providerModels = effectiveConfig.models[provider] || []
    if (providerModels.length === 0) return

    const isAvail = (m) => m.available !== false

    const storedModel = localStorage.getItem(STORAGE_KEYS.model)
    const stored = storedModel && providerModels.find(m => m.id === storedModel)
    if (stored && isAvail(stored)) {
      setModelId(stored.id)
      return
    }

    const firstAvailable = providerModels.find(isAvail)
    setModelId((firstAvailable || providerModels[0]).id)
  }, [userStudyMode, effectiveConfig, provider, baseline])

  // 切换提供商时，优先选择该提供商的第一个可用模型。
  const handleProviderChange = (newProvider) => {
    setProvider(newProvider)
    const providerModels = effectiveConfig?.models?.[newProvider] || []
    if (providerModels.length > 0) {
      const firstAvailable = providerModels.find(m => m.available !== false)
      setModelId((firstAvailable || providerModels[0]).id)
    }
    setValidationError(null)
  }

  // 处理 基线 更改 - 设置适当的默认
  const handleBaselineChange = (newBaseline) => {
    setBaseline(newBaseline)
    setValidationError(null)
    
    // PHIA不支持Anthropic(没有OneTwo后端).
    // 如果正在运行 Anthropic 切换到 PHIA 切换到 Gemini
    if (newBaseline === 'phia' && provider === 'anthropic') {
      setProvider('gemini')
      const geminiModels = effectiveConfig?.models?.gemini || []
      const pro25 = geminiModels.find(m => m.id.includes('gemini-2.5-pro'))
      if (pro25) {
        setModelId(pro25.id)
      } else if (geminiModels.length > 0) {
        setModelId(geminiModels[0].id)
      }
    }
  }

  // 确定根据所选选项需要哪个 API 密钥
   const getRequiredKeyType = () => {
    // 返回选中的提供者( PHIA现在既支持 Gemini 也支持 OpenAI )
    return provider
  }

  const requiredKeyType = getRequiredKeyType()
  const hasRequiredKey = userStudyMode ? true : (apiKeys[requiredKeyType]?.trim().length > 0)
  
  // 提供方选择
  const providerDisabled = false
  
  // 获取当前有效的提供者( 用于模型列表)
  const effectiveProvider = provider

  const keyInfo = {
    gemini: { label: 'Gemini', placeholder: 'AIza...', url: 'https://aistudio.google.com/apikey' },
    openai: { label: 'OpenAI', placeholder: 'sk-...', url: 'https://platform.openai.com/api-keys' },
    anthropic: { label: 'Anthropic', placeholder: 'sk-ant-...', url: 'https://console.anthropic.com/' },
    tavily: { label: 'Tavily', placeholder: 'tvly-...', url: 'https://tavily.com' },
  }

  const handleSubmit = (e) => {
    e.preventDefault()

    if (!hasRequiredKey) {
      setValidationError(`${keyInfo[requiredKeyType].label} API 密钥为必填项`)
      return
    }

    // 拦截已知不可用的模型，避免会话创建后第一次调用就失败。
    if (!userStudyMode) {
      const picked = currentModels.find(m => m.id === modelId)
      if (picked && picked.available === false) {
        setValidationError(
          `当前 ${effectiveProvider} 密钥无法访问「${picked.name}」，请选择可用模型。`
        )
        return
      }
    }

    setValidationError(null)

    // 构建会话配置
    const submittedProvider = userStudyMode ? (userStudyDefaults?.provider || provider) : provider
    const effectiveModelId = userStudyMode ? (userStudyDefaults?.model || modelId) : modelId
    const sessionConfig = {
      baseline,
      persona_id: persona,
      provider: submittedProvider,
      model_id: effectiveModelId,
      user_id: userId,
    }
    
    // 用户研究模式由后端配置密钥，普通模式使用表单中填写的密钥。
    if (!userStudyMode) {
      if (apiKeys.gemini) sessionConfig.gemini_api_key = apiKeys.gemini
      if (apiKeys.openai) sessionConfig.openai_api_key = apiKeys.openai
      if (apiKeys.anthropic) sessionConfig.anthropic_api_key = apiKeys.anthropic
      if (apiKeys.tavily) sessionConfig.tavily_api_key = apiKeys.tavily
    }
    
    onStart(sessionConfig)
  }

  const currentModels = effectiveConfig?.models?.[effectiveProvider] || []
  const selectedBaseline = config?.baselines?.find(b => b.id === baseline)
  const selectedPersona = config?.personas?.find(p => p.id === persona)

  return (
    <div className="splash">
      <div className="splash-container">
        <header className="splash-header">
          <h1>🏥 个人健康 Agent</h1>
          <p className="subtitle">PHA 研究门户</p>
        </header>

        <div className="pha-disclaimer" role="note">
          <span className="pha-disclaimer-label">提示：</span>
          本门户仅供研究使用，不作为医疗器械，也不适用于任何临床用途，包括诊断、预后判断和治疗建议。
        </div>

        <form onSubmit={handleSubmit} className="config-form">
          
          {/* 用户 ID 条目 */}
          <section className="config-section user-id-section">
            <h2>用户 ID</h2>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="请输入用户 ID"
              className="config-input user-id-input"
            />
          </section>

          {/* 步骤 1 : 基线 选择 */}
          <section className="config-section">
            <h2>1. 选择系统架构</h2>
            <div className="baseline-grid">
              {config?.baselines?.map(b => (
                <div
                  key={b.id}
                  className={`baseline-card ${baseline === b.id ? 'active' : ''}`}
                  onClick={() => handleBaselineChange(b.id)}
                >
                  <div className="baseline-header">
                    <input
                      type="radio"
                      name="baseline"
                      value={b.id}
                      checked={baseline === b.id}
                      onChange={() => handleBaselineChange(b.id)}
                    />
                    <span className="baseline-name">{b.name}</span>
                  </div>
                  <p className="baseline-desc">{b.description}</p>
                  <span className="baseline-arch">{b.architecture}</span>
                </div>
              ))}
            </div>
          </section>

          {/* 步骤 2: 提供者选择(隐藏在用户研究模式中) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>
              2. 选择模型提供商
            </h2>
            <div className="provider-grid">
              {config?.providers?.map(p => {
                const isAnthropic = p === 'anthropic'
                // Anthropic 支持 PHA 和 Parallel 架构。PHIA 使用的 OneTwo 后端不支持 Anthropic。
                const isDisabled = isAnthropic && baseline === 'phia'

                let disabledReason = ''
                if (isDisabled) {
                  disabledReason = 'PHIA 使用 OneTwo，目前无法调用 Anthropic 模型。使用 Anthropic 时请选择 PHA 或 Parallel。'
                }
                
                return (
                  <div key={p} className="provider-wrapper">
                    {isDisabled && disabledReason && (
                      <p className="provider-unavailable-note">
                        {disabledReason}
                      </p>
                    )}
                    <button
                      type="button"
                      className={`provider-btn ${effectiveProvider === p ? 'active' : ''} ${isDisabled ? 'coming-soon' : ''}`}
                      onClick={() => !isDisabled && handleProviderChange(p)}
                      disabled={isDisabled}
                    >
                      <span className="provider-icon">
                        {p === 'gemini' && '🔷'}
                        {p === 'openai' && '🟢'}
                        {p === 'anthropic' && '🟠'}
                      </span>
                      <span className="provider-name">
                        {keyInfo[p]?.label || p}
                      </span>
                    </button>
                  </div>
                )
              })}
            </div>
          </section>
          )}

          {/* 步骤 3:模型选择(隐藏在用户研究模式中) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>3. 选择模型</h2>
            <select
              value={modelId}
              onChange={(e) => {
                const next = e.target.value
                const picked = currentModels.find(m => m.id === next)
                if (picked && picked.available === false) return
                setModelId(next)
              }}
              className="config-select"
            >
              {currentModels.map(m => {
                const isUnavailable = m.available === false
                return (
                  <option
                    key={m.id}
                    value={m.id}
                    disabled={isUnavailable}
                    style={isUnavailable ? { color: '#94a3b8' } : undefined}
                  >
                    {m.name}{isUnavailable ? '（当前密钥无法访问）' : ''}
                  </option>
                )
              })}
            </select>
            {(() => {
              const hasProviderKey = !!apiKeys[effectiveProvider]
              if (!hasProviderKey) {
                return (
                  <p className="model-availability-note">
                    请在下方填写 {keyInfo[effectiveProvider]?.label || effectiveProvider} 密钥，查看账号可访问的模型。
                  </p>
                )
              }
              if (checkingAvailability) {
                return (
                  <p className="model-availability-note checking">
                    正在验证 {keyInfo[effectiveProvider]?.label || effectiveProvider} 模型的可用性…
                  </p>
                )
              }
              const total = currentModels.length
              const available = currentModels.filter(m => m.available !== false).length
              const allOk = available === total
              return (
                <p className={`model-availability-note ${allOk ? 'ok' : 'partial'}`}>
                  {allOk
                    ? `✓ 当前密钥可访问全部 ${total} 个模型。`
                    : `当前密钥可访问 ${total} 个模型中的 ${available} 个。`}
                </p>
              )
            })()}
          </section>
          )}

          {/* 步骤 4:人物选择 */}
          <section className="config-section">
            <h2>{userStudyMode ? '2. 选择用户档案' : '4. 选择用户档案'}</h2>
            <select
              value={persona}
              onChange={(e) => setPersona(e.target.value)}
              className="config-select"
            >
              {config?.personas?.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name} {p.description && `- ${p.description}`}
                </option>
              ))}
            </select>
            {selectedPersona?.demographics && (
              <div className="persona-details">
                {selectedPersona.demographics.age && <span>年龄：{selectedPersona.demographics.age} 岁</span>}
                {selectedPersona.demographics.gender && <span>性别：{{ male: '男', female: '女' }[String(selectedPersona.demographics.gender).toLowerCase()] ?? selectedPersona.demographics.gender}</span>}
                {selectedPersona.data_summary?.days && <span>{selectedPersona.data_summary.days} 天的数据</span>}
              </div>
            )}
          </section>

          {/* 步 5: API 键(隐藏在用户研究模式中) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>5. API 密钥</h2>
            
            {/* 需要的 API 密钥 */}
            <div className="api-key-group">
              <h3 className="api-key-title">
                <span className="required-dot"></span>
                必填：{keyInfo[requiredKeyType].label} API 密钥
              </h3>
              <p className="api-key-reason">
                {`调用 ${keyInfo[requiredKeyType].label} 模型时需要此密钥`}
              </p>
              <input
                type="password"
                placeholder={keyInfo[requiredKeyType].placeholder}
                value={apiKeys[requiredKeyType]}
                onChange={(e) => setApiKeys({...apiKeys, [requiredKeyType]: e.target.value})}
                className={`api-key-input ${validationError && !hasRequiredKey ? 'error' : ''}`}
              />
              <a 
                href={keyInfo[requiredKeyType].url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="api-key-link"
              >
                获取 {keyInfo[requiredKeyType].label} API 密钥 →
              </a>
            </div>
            
            {/* 可选 Tavily 密钥 */}
            <div className="api-key-group optional">
              <h3 className="api-key-title">
                <span className="optional-dot"></span>
                选填：Tavily API 密钥
              </h3>
              <p className="api-key-reason">供领域专家 Agent 搜索健康资料</p>
              <input
                type="password"
                placeholder="tvly-..."
                value={apiKeys.tavily}
                onChange={(e) => setApiKeys({...apiKeys, tavily: e.target.value})}
                className="api-key-input"
              />
              <a 
                href="https://tavily.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="api-key-link"
              >
                获取 Tavily API 密钥 →
              </a>
            </div>
            
            <p className="api-env-hint">
              💡 也可通过服务端环境变量配置密钥
              {(apiKeys.gemini || apiKeys.openai || apiKeys.anthropic || apiKeys.tavily) && (
                <span className="saved-indicator"> • ✓ 密钥已保存在本机</span>
              )}
            </p>
          </section>
          )}

          {/* 显示错误 */}
          {(validationError || error) && (
            <div className="config-error">
              ⚠️ {validationError || error}
            </div>
          )}

          {/* 启动按钮 */}
          <button type="submit" className="start-btn" disabled={loading}>
            {loading ? (
              <>
                <span className="btn-spinner"></span>
                正在创建会话…
              </>
            ) : (
              '开始对话 →'
            )}
          </button>

          {/* 概要 */}
          <div className="config-summary">
            <span>{selectedBaseline?.name}</span>
            <span>•</span>
            <span>{modelId ? modelId.split('/').pop() : '—'}</span>
            <span>•</span>
            <span>{selectedPersona?.name}</span>
          </div>
        </form>

        <footer className="splash-footer">
          <p>PHA 可以协同多个 Agent 分析个人健康数据</p>
          <button
            type="button" 
            className="clear-data-btn"
            onClick={() => {
              Object.values(STORAGE_KEYS).forEach(key => localStorage.removeItem(key))
              setApiKeys({ gemini: '', openai: '', anthropic: '', tavily: '' })
              setProvider('gemini')
              setBaseline('pha')
              setPersona('sample')
              setUserId('test')
            }}
          >
            清除本机保存的设置
          </button>
        </footer>
      </div>
    </div>
  )
}

export default SplashScreen

import React, { useState, useEffect, useMemo, useRef } from 'react'
import './SplashScreen.css'

const API_BASE = '/api'

// localStorage 密钥
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

  // 在用户学习模式下获取用户学习默认值
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

  // 任何 API 关键更改时重新检查模型可用性。
  // -在第一舱： 立即开火，所以节省的钥匙在前面被验证。
  // - 之后的关键改变： 卸下500米这样我们就不会在每次键盘中取回。
  // -如果没有输入密钥，则不做任何动作(保留所有可用默认的全程目录)。
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

  // 当配置负载或提供者改变时初始化模型(在用户学习模式下进行skip).
  // 优惠顺序:如果有的话，存储模型;第一个可用模型;第一个模型。
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

  // 当提供者改变时更新模型——先选择可用，或先作为倒置。
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
      setValidationError(`${keyInfo[requiredKeyType].label} API key is required`)
      return
    }

    // 如果已知选定模型无法使用，则块提交
    // 键。这阻止用户踢出聊天会话
    // 第一次会失败 LLM 打电话。
    if (!userStudyMode) {
      const picked = currentModels.find(m => m.id === modelId)
      if (picked && picked.available === false) {
        setValidationError(
          `"${picked.name}" is not available for this ${effectiveProvider} key. Pick an enabled option.`
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
    
    // API 密钥:在用户学习模式后端从配置文件注入;否则从形式中注入
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
          <h1>🏥 PHA Portal</h1>
          <p className="subtitle">Personal Health Agent</p>
        </header>

        <div className="pha-disclaimer" role="note">
          <span className="pha-disclaimer-label">NOTE:</span>
          This research portal (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis, prognosis, or treatment recommendations.
        </div>

        <form onSubmit={handleSubmit} className="config-form">
          
          {/* 用户 ID 条目 */}
          <section className="config-section user-id-section">
            <h2>User ID</h2>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter your user ID"
              className="config-input user-id-input"
            />
          </section>

          {/* 步骤 1 : 基线 选择 */}
          <section className="config-section">
            <h2>1. Select System Architecture</h2>
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

          {/* 步骤 2: 提供者选择(隐藏在用户学习模式中) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>
              2. Select Provider
            </h2>
            <div className="provider-grid">
              {config?.providers?.map(p => {
                const isAnthropic = p === 'anthropic'
                // Anthropic由PHA和并行基线(多型号)支持。
                // 后端),但不由PHIA(使用OneTwo/没有Anthropic后端).
                const isDisabled = isAnthropic && baseline === 'phia'

                let disabledReason = ''
                if (isDisabled) {
                  disabledReason = 'Anthropic is not supported by the PHIA baseline (OneTwo has no Anthropic backend). Use PHA or Parallel.'
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
                        {p.charAt(0).toUpperCase() + p.slice(1)}
                      </span>
                    </button>
                  </div>
                )
              })}
            </div>
          </section>
          )}

          {/* 步骤 3:模型选择(隐藏在用户学习模式中) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>3. Select Model</h2>
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
                    {m.name}{isUnavailable ? ' — not available for this key' : ''}
                  </option>
                )
              })}
            </select>
            {(() => {
              const hasProviderKey = !!apiKeys[effectiveProvider]
              if (!hasProviderKey) {
                return (
                  <p className="model-availability-note">
                    Enter a {effectiveProvider.charAt(0).toUpperCase() + effectiveProvider.slice(1)} key below to verify which models your account can access.
                  </p>
                )
              }
              if (checkingAvailability) {
                return (
                  <p className="model-availability-note checking">
                    Verifying availability with {effectiveProvider}…
                  </p>
                )
              }
              const total = currentModels.length
              const available = currentModels.filter(m => m.available !== false).length
              const allOk = available === total
              return (
                <p className={`model-availability-note ${allOk ? 'ok' : 'partial'}`}>
                  {allOk
                    ? `✓ All ${total} models available for this ${effectiveProvider} key.`
                    : `${available} of ${total} models available for this ${effectiveProvider} key.`}
                </p>
              )
            })()}
          </section>
          )}

          {/* 步骤 4:人物选择 */}
          <section className="config-section">
            <h2>{userStudyMode ? '2. Select User Profile' : '4. Select User Profile'}</h2>
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
                {selectedPersona.demographics.age && <span>Age: {selectedPersona.demographics.age}</span>}
                {selectedPersona.demographics.gender && <span>Gender: {selectedPersona.demographics.gender}</span>}
                {selectedPersona.data_summary?.days && <span>{selectedPersona.data_summary.days} days of data</span>}
              </div>
            )}
          </section>

          {/* 步 5: API 键(隐藏在用户学习模式中) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>5. API Keys</h2>
            
            {/* 需要的 API 密钥 */}
            <div className="api-key-group">
              <h3 className="api-key-title">
                <span className="required-dot"></span>
                Required: {keyInfo[requiredKeyType].label} API Key
              </h3>
              <p className="api-key-reason">
                {`Required for ${provider.charAt(0).toUpperCase() + provider.slice(1)} models`}
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
                Get {keyInfo[requiredKeyType].label} API Key →
              </a>
            </div>
            
            {/* 可选 Tavily 密钥 */}
            <div className="api-key-group optional">
              <h3 className="api-key-title">
                <span className="optional-dot"></span>
                Optional: Tavily API Key
              </h3>
              <p className="api-key-reason">Enables web search for health information lookups</p>
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
                Get Tavily API Key →
              </a>
            </div>
            
            <p className="api-env-hint">
              💡 Keys can also be set as environment variables on the server
              {(apiKeys.gemini || apiKeys.openai || apiKeys.anthropic || apiKeys.tavily) && (
                <span className="saved-indicator"> • ✓ Keys saved locally</span>
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
                Initializing...
              </>
            ) : (
              'Start Chat Session →'
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
          <p>PHA - A multi-agent system for personal health insights</p>
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
            Clear saved data
          </button>
        </footer>
      </div>
    </div>
  )
}

export default SplashScreen

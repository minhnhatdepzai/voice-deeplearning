import React, { useEffect, useRef, useState } from 'react'
import { AudioLines, Gauge, Mic2, Play, Radio, Square, Zap } from 'lucide-react'

const phaseLabels = {
  idle: 'Sẵn sàng thu', warming: 'Đang nạp model một lần', recording: 'Đang thu liên tục',
  processing: 'Đang nhận dạng và đổi giọng', playing: 'Đang phát giọng mới',
}

function recorderOptions() {
  if (!window.MediaRecorder) throw new Error('Trình duyệt này chưa hỗ trợ ghi âm MediaRecorder.')
  const candidates = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm']
  const mimeType = candidates.find(type => MediaRecorder.isTypeSupported(type))
  return mimeType ? { mimeType, audioBitsPerSecond: 64000 } : { audioBitsPerSecond: 64000 }
}

function formatDuration(value) {
  const total = Math.max(0, Math.floor(Number(value) || 0))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const seconds = total % 60
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}` : `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

export default function SpeechToSpeech({ api, voices, onError, onNotice }) {
  const trained = voices.filter(voice => voice.kind === 'finetuned')
  const personalTrained = trained.filter(voice => voice.source === 'upload')
  const [voiceId, setVoiceId] = useState('')
  const [language, setLanguage] = useState('vi')
  const [quality, setQuality] = useState('balanced')
  const [denoise, setDenoise] = useState('light')
  const [phase, setPhase] = useState('idle')
  const [level, setLevel] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [history, setHistory] = useState([])
  const [warmup, setWarmup] = useState(null)
  const [job, setJob] = useState(null)
  const runtime = useRef({})
  const runId = useRef(0)

  useEffect(() => {
    if (!voices.some(voice => voice.id === voiceId)) setVoiceId(personalTrained.find(voice => voice.s2s_recommended)?.id || personalTrained[0]?.id || trained[0]?.id || voices[0]?.id || '')
  }, [voices, voiceId])

  function releaseCapture() {
    const state = runtime.current
    if (state.frame) cancelAnimationFrame(state.frame)
    state.source?.disconnect(); state.analyser?.disconnect(); state.silent?.disconnect()
    state.stream?.getTracks().forEach(track => track.stop())
    state.context?.close().catch(() => {})
    state.stream = null; state.context = null; state.source = null; state.analyser = null; state.silent = null
    setLevel(0)
  }

  function releaseBrowser() {
    const state = runtime.current
    if (state.recorder?.state === 'recording') {
      state.recorder.onstop = null
      state.recorder.stop()
    }
    releaseCapture()
    state.player?.pause()
    runtime.current = {}
    setElapsed(0)
  }

  async function cancel() {
    runId.current += 1
    releaseBrowser()
    setPhase('idle')
    await api('/speech-to-speech/release', { method: 'POST' }).catch(() => {})
  }

  useEffect(() => () => {
    runId.current += 1
    releaseBrowser()
  }, [])

  useEffect(() => {
    const saved = localStorage.getItem('vocalab-speech-job')
    if (saved) { setJob({ id: saved, status: 'queued', stage: 'queued' }); setPhase('processing') }
  }, [])

  useEffect(() => {
    if (!job?.id || job.status === 'complete' || job.status === 'failed') return
    let active = true
    const poll = async () => {
      try {
        const state = await api(`/speech-to-speech/jobs/${job.id}`)
        if (!active) return
        setJob(state)
        if (state.status === 'complete') {
          localStorage.removeItem('vocalab-speech-job')
          const result = state.result
          setHistory(previous => [{ ...result, id: state.id, at: new Date().toLocaleTimeString('vi-VN') }, ...previous].slice(0, 12))
          const player = new Audio(result.audio_url)
          runtime.current.player = player
          setPhase('playing')
          const finish = () => { runtime.current.player = null; setPhase('idle') }
          player.onended = finish; player.onerror = finish
          player.play().catch(finish)
        } else if (state.status === 'failed') {
          localStorage.removeItem('vocalab-speech-job')
          onError(state.error || 'Không thể đổi giọng bản ghi.')
          setPhase('idle')
        }
      } catch (error) {
        if (active) { onError(error.message); setPhase('idle'); localStorage.removeItem('vocalab-speech-job') }
      }
    }
    poll()
    const timer = setInterval(poll, 2000)
    return () => { active = false; clearInterval(timer) }
  }, [job?.id, job?.status])

  function monitorInput(id) {
    const state = runtime.current
    if (id !== runId.current || !state.analyser) return
    const samples = new Float32Array(state.analyser.fftSize)
    state.analyser.getFloatTimeDomainData(samples)
    let power = 0
    for (const sample of samples) power += sample * sample
    setLevel(Math.min(1, Math.sqrt(power / samples.length) * 12))
    if (state.startedAt) setElapsed((performance.now() - state.startedAt) / 1000)
    state.frame = requestAnimationFrame(() => monitorInput(id))
  }

  async function submitRecording(blob, mimeType, id) {
    try {
      if (blob.size < 512) throw new Error('Bản ghi chưa có đủ âm thanh để xử lý.')
      const extension = mimeType.includes('ogg') ? 'ogg' : 'webm'
      const form = new FormData()
      form.append('file', new File([blob], `speech-${Date.now()}.${extension}`, { type: mimeType }))
      form.append('voice_id', voiceId); form.append('language', language); form.append('quality', quality)
      form.append('denoise', denoise)
      const result = await api('/speech-to-speech/jobs', { method: 'POST', body: form })
      if (id !== runId.current) return
      localStorage.setItem('vocalab-speech-job', result.id)
      setJob(result)
    } catch (error) {
      if (id === runId.current) { onError(error.message); setPhase('idle') }
    }
  }

  function stopAndListen() {
    const state = runtime.current
    if (phase !== 'recording' || state.recorder?.state !== 'recording') return
    setPhase('processing')
    state.recorder.stop()
  }

  async function start() {
    if (!voiceId) return onError('Hãy chọn một giọng đã train trước.')
    onError(''); onNotice(''); setElapsed(0)
    const id = ++runId.current
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      } })
      const context = new AudioContext({ latencyHint: 'interactive' })
      await context.resume()
      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser(); analyser.fftSize = 2048
      const silent = context.createGain(); silent.gain.value = 0
      source.connect(analyser); analyser.connect(silent); silent.connect(context.destination)
      const recorder = new MediaRecorder(stream, recorderOptions())
      const chunks = []
      runtime.current = { stream, context, source, analyser, silent, recorder, chunks, startedAt: 0, frame: 0 }
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data) }
      recorder.onerror = event => { if (id === runId.current) { onError(event.error?.message || 'Không thể tiếp tục ghi âm.'); cancel() } }
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || 'audio/webm'
        const blob = new Blob(chunks, { type: mimeType })
        runtime.current.chunks = []
        runtime.current.recorder = null
        releaseCapture()
        if (id === runId.current) submitRecording(blob, mimeType, id)
      }
      setPhase('warming')
      const state = await api('/speech-to-speech/warmup', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId, quality, language }) })
      if (id !== runId.current) return
      setWarmup(state)
      runtime.current.startedAt = performance.now()
      recorder.start(1000)
      setPhase('recording')
      monitorInput(id)
      onNotice('Đang thu liên tục. Bạn có thể nói và nghỉ tùy ý; khi hoàn tất hãy bấm “Dừng & nghe”.')
    } catch (error) {
      if (id === runId.current) { releaseBrowser(); setPhase('idle'); onError(error.message) }
    }
  }

  function stopPlayback() {
    runtime.current.player?.pause()
    runtime.current.player = null
    setPhase('idle')
  }

  const active = phase !== 'idle'
  const button = phase === 'idle'
    ? { action: start, disabled: !voiceId, icon: <Mic2 size={18}/>, text: 'Bắt đầu thu' }
    : phase === 'recording'
      ? { action: stopAndListen, disabled: false, icon: <Square size={17}/>, text: 'Dừng & nghe' }
      : phase === 'warming'
        ? { action: cancel, disabled: false, icon: <Square size={17}/>, text: 'Hủy' }
        : phase === 'playing'
          ? { action: stopPlayback, disabled: false, icon: <Square size={17}/>, text: 'Dừng phát' }
          : { action: () => {}, disabled: true, icon: <Zap size={18}/>, text: 'Đang xử lý bản ghi' }

  return <>
    <div className="page-heading compact"><div><div className="eyebrow"><span/> SPEECH TO SPEECH · THU THỦ CÔNG</div><h1>Nói đầy đủ. Bấm dừng. <em>AI mới bắt đầu đổi giọng.</em></h1><p>Không tự ngắt khi bạn im lặng và không giới hạn số giây mỗi lượt. Toàn bộ bản ghi chỉ được xử lý sau khi bạn bấm Dừng & nghe.</p></div></div>
    <div className="s2s-grid">
      <section className="panel s2s-control">
        <div className="panel-title"><span className="panel-icon purple"><Radio size={20}/></span><div><h3>Thu và đổi giọng toàn đoạn</h3><p>Model được giữ nóng để lần tiếp theo nhanh hơn</p></div></div>
        <label className="field-label" htmlFor="s2s-voice">Giọng đầu ra</label>
        <div className="select-wrap"><select id="s2s-voice" value={voiceId} disabled={active} onChange={event => setVoiceId(event.target.value)}>
          {voices.map(voice => <option value={voice.id} key={voice.id}>{voice.name} · {voice.kind === 'finetuned' ? 'đã train' : 'mẫu tham chiếu'}</option>)}
        </select></div>
        <div className="s2s-options"><label><span>Ngôn ngữ nói</span><select value={language} disabled={active} onChange={event => setLanguage(event.target.value)}><option value="vi">Tiếng Việt</option><option value="en">English</option><option value="auto">Tự nhận diện</option></select></label><label><span>Độ chính xác</span><select value={quality} disabled={active} onChange={event => setQuality(event.target.value)}><option value="fast">Nhanh · Whisper small beam 1</option><option value="balanced">Cân bằng · Whisper turbo beam 3</option><option value="accurate">Chuẩn hơn · Whisper turbo beam 5</option></select></label><label><span>Lọc tiếng ù đầu ra</span><select value={denoise} disabled={active} onChange={event => setDenoise(event.target.value)}><option value="light">Nhẹ · giữ tự nhiên</option><option value="strong">Mạnh · ít nền hơn</option><option value="off">Tắt lọc</option></select></label></div>
        <div className={`s2s-orb ${phase}`}><div className="s2s-level" style={{ transform: `scale(${1 + level * .32})` }}/>{phase === 'recording' ? <AudioLines size={46}/> : phase === 'processing' || phase === 'warming' ? <Zap size={44}/> : <Mic2 size={46}/>}</div>
        <div className="s2s-state"><span className={`state-dot ${active ? 'live' : ''}`}/><strong>{phaseLabels[phase]}</strong>{phase === 'recording' && <small>Đã thu {formatDuration(elapsed)} · chỉ xử lý khi bấm Dừng & nghe</small>}{phase === 'processing' && <small>{({ queued: 'Đang chờ', decoding: 'Đang giải mã', transcribing: 'Đang nhận dạng lời nói', synthesizing: 'Đang tạo giọng' })[job?.stage] || 'Đang tải bản ghi'}{job?.chunks_total ? ` · ${job.chunks_done || 0}/${job.chunks_total} đoạn` : ''} · có thể tải lại trang để xem tiếp</small>}</div>
        <button type="button" className={`button full ${active ? 'outline danger' : 'primary'}`} onClick={button.action} disabled={button.disabled}>{button.icon} {button.text}</button>
        <p className="fine-print">Không có bộ đếm tự dừng. Giới hạn thực tế phụ thuộc bộ nhớ trình duyệt và dung lượng tải lên tối đa 500 MB. Đeo tai nghe để tránh micro thu lại âm thanh từ loa.</p>
      </section>
      <section className="panel s2s-history">
        <div className="panel-title"><span className="panel-icon blue"><Gauge size={20}/></span><div><h3>Bản ghi vừa chuyển</h3><p>Phiên âm, file giọng mới và thời gian xử lý thật</p></div></div>
        {warmup && <div className="s2s-warm"><Zap size={16}/><span>{warmup.stt_model} · {warmup.stt_device}</span><strong>Nạp {(warmup.warmup_ms / 1000).toFixed(1)}s</strong></div>}
        {history.length ? <div className="s2s-results">{history.map(item => <article key={item.id}><div><time>{item.at}</time><span>{item.language.toUpperCase()} · bản ghi {formatDuration(item.input_seconds)} · {item.synthesis_chunks || 1} đoạn tạo giọng</span></div><p>{item.text}</p><div className="s2s-metrics"><span>STT {(item.latency_ms.stt / 1000).toFixed(2)}s</span><span>TTS {(item.latency_ms.tts / 1000).toFixed(2)}s</span><strong>Tổng {(item.latency_ms.total / 1000).toFixed(2)}s</strong><button type="button" onClick={() => new Audio(item.audio_url).play()}><Play size={14}/> Nghe lại</button></div></article>)}</div> : <div className="empty-result"><div className="empty-orbit"><AudioLines size={40}/></div><h3>Chưa có bản ghi nào</h3><p>Bấm Bắt đầu thu, nói đầy đủ rồi bấm Dừng & nghe.</p></div>}
      </section>
    </div>
  </>
}

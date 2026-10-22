import React, { useEffect, useRef, useState } from 'react'
import { AudioLines, Gauge, Mic2, Pause, Play, Radio, Zap } from 'lucide-react'

function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const buffer = new ArrayBuffer(44 + length * 2)
  const view = new DataView(buffer)
  const write = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)))
  write(0, 'RIFF'); view.setUint32(4, 36 + length * 2, true); write(8, 'WAVE')
  write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  write(36, 'data'); view.setUint32(40, length * 2, true)
  let offset = 44
  for (const chunk of chunks) for (const sample of chunk) {
    const clipped = Math.max(-1, Math.min(1, sample))
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true)
    offset += 2
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

const phaseLabels = {
  idle: 'Chưa bật micro', warming: 'Đang nạp model một lần', listening: 'Đang nghe',
  speaking: 'Đang thu câu nói', processing: 'Đang nhận dạng và đổi giọng', playing: 'Đang phát giọng mới',
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
  const [history, setHistory] = useState([])
  const [warmup, setWarmup] = useState(null)
  const runtime = useRef({})
  const runId = useRef(0)

  useEffect(() => {
    if (!voices.some(voice => voice.id === voiceId)) setVoiceId(personalTrained.find(voice => voice.s2s_recommended)?.id || personalTrained[0]?.id || trained[0]?.id || voices[0]?.id || '')
  }, [voices, voiceId])

  function releaseBrowser() {
    const state = runtime.current
    state.processor?.disconnect(); state.source?.disconnect(); state.silent?.disconnect()
    state.stream?.getTracks().forEach(track => track.stop())
    state.context?.close().catch(() => {})
    state.player?.pause()
    runtime.current = {}
    setLevel(0)
  }

  async function stop() {
    runId.current += 1
    releaseBrowser()
    setPhase('idle')
    await api('/speech-to-speech/release', { method: 'POST' }).catch(() => {})
  }

  useEffect(() => () => {
    runId.current += 1
    releaseBrowser()
    fetch('/api/speech-to-speech/release', { method: 'POST' }).catch(() => {})
  }, [])

  function resumeListening(id) {
    if (id !== runId.current || !runtime.current.context) return
    runtime.current.active = false
    runtime.current.utterance = []
    runtime.current.preRoll = []
    runtime.current.speechFrames = 0
    runtime.current.lastVoiceAt = 0
    setPhase('listening')
  }

  async function submitUtterance(chunks, sampleRate, id) {
    try {
      const blob = encodeWav(chunks, sampleRate)
      const form = new FormData()
      form.append('file', new File([blob], `utterance-${Date.now()}.wav`, { type: 'audio/wav' }))
      form.append('voice_id', voiceId); form.append('language', language); form.append('quality', quality)
      form.append('denoise', denoise)
      const result = await api('/speech-to-speech', { method: 'POST', body: form })
      if (id !== runId.current) return
      setHistory(previous => [{ ...result, id: crypto.randomUUID(), at: new Date().toLocaleTimeString('vi-VN') }, ...previous].slice(0, 12))
      const player = new Audio(result.audio_url)
      runtime.current.player = player
      setPhase('playing')
      player.onended = () => resumeListening(id)
      player.onerror = () => resumeListening(id)
      await player.play().catch(() => resumeListening(id))
    } catch (error) {
      if (id === runId.current) {
        onError(error.message)
        resumeListening(id)
      }
    }
  }

  function finishUtterance(id) {
    const state = runtime.current
    if (!state.active || !state.utterance?.length || id !== runId.current) return
    const chunks = state.utterance
    state.active = false; state.utterance = []; state.preRoll = []
    setPhase('processing')
    submitUtterance(chunks, state.context.sampleRate, id)
  }

  function handleAudio(event, id) {
    const state = runtime.current
    if (!state.context || id !== runId.current || !['listening', 'speaking'].includes(state.phase)) return
    const chunk = new Float32Array(event.inputBuffer.getChannelData(0))
    let power = 0
    for (const sample of chunk) power += sample * sample
    const rms = Math.sqrt(power / chunk.length)
    if (!state.lastLevelAt || performance.now() - state.lastLevelAt > 70) {
      state.lastLevelAt = performance.now(); setLevel(Math.min(1, rms * 12))
    }
    if (!state.active) {
      // Only learn from quiet frames. If the user speaks immediately after
      // enabling the mic, their voice must not become the ambient baseline.
      if (rms < Math.max(0.018, state.noise * 2.5)) state.noise = state.noise * 0.98 + rms * 0.02
      state.preRoll.push(chunk)
      state.preRollSamples += chunk.length
      while (state.preRollSamples > state.context.sampleRate * 0.35 && state.preRoll.length > 1) {
        state.preRollSamples -= state.preRoll[0].length; state.preRoll.shift()
      }
    }
    const threshold = Math.max(0.012, (state.noise || 0.004) * 2.7)
    if (rms > threshold) state.speechFrames += 1
    else state.speechFrames = Math.max(0, state.speechFrames - 1)
    const now = performance.now()
    if (!state.active && state.speechFrames >= 2) {
      state.active = true; state.startedAt = now; state.lastVoiceAt = now
      state.utterance = [...state.preRoll]; state.preRoll = []; state.preRollSamples = 0
      state.phase = 'speaking'; setPhase('speaking')
    } else if (state.active) {
      state.utterance.push(chunk)
      if (rms > threshold) state.lastVoiceAt = now
      if ((now - state.lastVoiceAt > 680 && now - state.startedAt > 350) || now - state.startedAt > 15000) finishUtterance(id)
    }
  }

  async function start() {
    if (!voiceId) return onError('Hãy chọn một giọng đã train trước.')
    onError(''); onNotice('')
    const id = ++runId.current
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      } })
      const context = new AudioContext({ latencyHint: 'interactive' })
      await context.resume()
      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(2048, 1, 1)
      const silent = context.createGain(); silent.gain.value = 0
      runtime.current = { stream, context, source, processor, silent, phase: 'warming', active: false,
        utterance: [], preRoll: [], preRollSamples: 0, speechFrames: 0, noise: 0.003 }
      source.connect(processor); processor.connect(silent); silent.connect(context.destination)
      processor.onaudioprocess = event => handleAudio(event, id)
      setPhase('warming')
      const state = await api('/speech-to-speech/warmup', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId, quality, language }) })
      if (id !== runId.current) return
      setWarmup(state)
      runtime.current.phase = 'listening'; setPhase('listening')
      onNotice(`Model đã sẵn sàng sau ${(state.warmup_ms / 1000).toFixed(1)} giây. Hãy nói tự nhiên rồi ngừng khoảng 0,7 giây.`)
    } catch (error) {
      if (id === runId.current) { releaseBrowser(); setPhase('idle'); onError(error.message) }
    }
  }

  useEffect(() => { if (runtime.current.context) runtime.current.phase = phase }, [phase])
  const active = phase !== 'idle'

  return <>
    <div className="page-heading compact"><div><div className="eyebrow"><span/> SPEECH TO SPEECH · TỰ ĐỘNG</div><h1>Bạn nói. AI trả lại <em>đúng giọng đã train.</em></h1><p>Micro tự nhận biết khi bạn nói xong, phiên âm câu nói rồi phát ngay bằng giọng đã chọn.</p></div></div>
    <div className="s2s-grid">
      <section className="panel s2s-control">
        <div className="panel-title"><span className="panel-icon purple"><Radio size={20}/></span><div><h3>Phiên đổi giọng trực tiếp</h3><p>Model được giữ nóng trong suốt phiên</p></div></div>
        <label className="field-label" htmlFor="s2s-voice">Giọng đầu ra</label>
        <div className="select-wrap"><select id="s2s-voice" value={voiceId} disabled={active} onChange={event => setVoiceId(event.target.value)}>
          {voices.map(voice => <option value={voice.id} key={voice.id}>{voice.name} · {voice.kind === 'finetuned' ? 'đã train' : 'mẫu tham chiếu'}</option>)}
        </select></div>
        <div className="s2s-options"><label><span>Ngôn ngữ nói</span><select value={language} disabled={active} onChange={event => setLanguage(event.target.value)}><option value="vi">Tiếng Việt</option><option value="en">English</option><option value="auto">Tự nhận diện</option></select></label><label><span>Độ chính xác</span><select value={quality} disabled={active} onChange={event => setQuality(event.target.value)}><option value="fast">Nhanh · Whisper small beam 1</option><option value="balanced">Cân bằng · Whisper turbo beam 3</option><option value="accurate">Chuẩn hơn · Whisper turbo beam 5</option></select></label><label><span>Lọc tiếng ù đầu ra</span><select value={denoise} disabled={active} onChange={event => setDenoise(event.target.value)}><option value="light">Nhẹ · giữ tự nhiên</option><option value="strong">Mạnh · ít nền hơn</option><option value="off">Tắt lọc</option></select></label></div>
        <div className={`s2s-orb ${phase}`}><div className="s2s-level" style={{ transform: `scale(${1 + level * .32})` }}/>{phase === 'speaking' ? <AudioLines size={46}/> : phase === 'processing' || phase === 'warming' ? <Zap size={44}/> : <Mic2 size={46}/>}</div>
        <div className="s2s-state"><span className={`state-dot ${active ? 'live' : ''}`}/><strong>{phaseLabels[phase]}</strong>{phase === 'listening' && <small>Nói một câu; hệ thống tự chốt sau 0,7 giây im lặng</small>}{phase === 'speaking' && <small>Đang thu · tối đa 15 giây mỗi lượt</small>}</div>
        <button type="button" className={`button full ${active ? 'outline danger' : 'primary'}`} onClick={active ? stop : start} disabled={!voiceId}>{active ? <Pause size={18}/> : <Mic2 size={18}/>} {active ? 'Dừng phiên và giải phóng model' : 'Bật micro và nạp model'}</button>
        <p className="fine-print">Đeo tai nghe để tránh micro thu lại tiếng vừa phát. Lượt đầu dùng để nạp model; các lượt sau dùng cùng model trong VRAM nên nhanh hơn.</p>
      </section>
      <section className="panel s2s-history">
        <div className="panel-title"><span className="panel-icon blue"><Gauge size={20}/></span><div><h3>Câu vừa chuyển</h3><p>Phiên âm, file giọng mới và thời gian xử lý thật</p></div></div>
        {warmup && <div className="s2s-warm"><Zap size={16}/><span>{warmup.stt_model} · {warmup.stt_device}</span><strong>Nạp {(warmup.warmup_ms / 1000).toFixed(1)}s</strong></div>}
        {history.length ? <div className="s2s-results">{history.map(item => <article key={item.id}><div><time>{item.at}</time><span>{item.language.toUpperCase()} · câu vào {item.input_seconds}s</span></div><p>{item.text}</p><div className="s2s-metrics"><span>STT {(item.latency_ms.stt / 1000).toFixed(2)}s</span><span>TTS {(item.latency_ms.tts / 1000).toFixed(2)}s</span><strong>Tổng {(item.latency_ms.total / 1000).toFixed(2)}s</strong><button type="button" onClick={() => new Audio(item.audio_url).play()}><Play size={14}/> Nghe lại</button></div></article>)}</div> : <div className="empty-result"><div className="empty-orbit"><AudioLines size={40}/></div><h3>Chưa có câu nào</h3><p>Bật micro, chờ model sẵn sàng rồi nói một câu ngắn.</p></div>}
      </section>
    </div>
  </>
}

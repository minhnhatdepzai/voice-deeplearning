import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { AudioLines, ChevronDown, CircleHelp, Clock3, Cpu, Database, Download, FileAudio2, FileText, Film, FolderOpen, Headphones, Mic2, MoreHorizontal, Pause, Play, Plus, Radio, Scissors, Sparkles, Trash2, UploadCloud, Video, WandSparkles, X } from 'lucide-react'
import './style.css'
import Editor from './Editor.jsx'
import SpeechToSpeech from './SpeechToSpeech.jsx'

async function api(path, options) {
  const response = await fetch(`/api${path}`, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || `Lỗi HTTP ${response.status}`)
  return data
}

function useRecorder(onDone) {
  const recorder = useRef(null)
  const stream = useRef(null)
  const [recording, setRecording] = useState(false)
  async function toggle() {
    if (recording) { recorder.current?.stop(); return }
    const media = await navigator.mediaDevices.getUserMedia({ audio: true })
    stream.current = media
    const chunks = []
    const r = new MediaRecorder(media)
    recorder.current = r
    r.ondataavailable = e => { if (e.data.size) chunks.push(e.data) }
    r.onstop = () => {
      onDone(new File(chunks, 'ban-ghi-cua-toi.webm', { type: r.mimeType || 'audio/webm' }))
      media.getTracks().forEach(t => t.stop())
      setRecording(false)
    }
    r.start()
    setRecording(true)
  }
  useEffect(() => () => { recorder.current?.state === 'recording' && recorder.current.stop(); stream.current?.getTracks().forEach(t => t.stop()) }, [])
  return { recording, toggle }
}

const iconSize = 18
const bars = [18,30,22,38,19,43,32,15,27,44,33,21,48,28,17,35,42,22,31,48,23,36,14,30,45,26,18,40,28,21,34,16,42,27,19,38,25,15,31,40,20,36,16,28,43,21,33,15]

function Waveform({ active = false }) { return <div className={`waveform ${active ? 'active' : ''}`} aria-hidden="true">{bars.map((h, i) => <i key={i} style={{ height: `${h}px`, animationDelay: `${i * 37}ms` }} />)}</div> }

function FilePick({ file, onFile, accept, label = 'Chọn tệp từ máy', id, hint = 'Tối đa 500 MB · MP3, MP4 và các định dạng thông dụng' }) {
  const input = useRef(null)
  useEffect(() => { if (!file && input.current) input.current.value = '' }, [file])
  return <div className="file-pick-wrap"><label className={`file-pick ${file ? 'has-file' : ''}`} htmlFor={id}>
    <input ref={input} id={id} type="file" accept={accept} onChange={e => onFile(e.target.files?.[0] || null)} />
    <div className="file-icon">{file ? <FileAudio2 size={22}/> : <UploadCloud size={24}/>}</div>
    <div><strong>{file ? file.name : label}</strong><span>{file ? `${(file.size / 1048576).toFixed(1)} MB · Bấm để đổi tệp` : hint}</span></div>
    <span className="pick-button">Duyệt tệp</span>
  </label>{file && <button type="button" className="file-clear" onClick={() => { onFile(null); if (input.current) input.current.value = '' }} aria-label={`Bỏ file ${file.name}`}><X size={15}/> Bỏ file</button>}</div>
}

function SourcePreview({ file }) {
  const [url, setUrl] = useState('')
  useEffect(() => { if (!file) { setUrl(''); return } const objectUrl = URL.createObjectURL(file); setUrl(objectUrl); return () => URL.revokeObjectURL(objectUrl) }, [file])
  if (!file || !url) return null
  const isVideo = file.type.startsWith('video/') || (!file.type && /\.(mp4|mov|mkv)$/i.test(file.name))
  return <div className="source-preview"><strong>Xem toàn bộ file nguồn</strong>{isVideo ? <video src={url} controls preload="metadata"/> : <audio src={url} controls preload="metadata"/>}</div>
}

function minutes(seconds) { return `${(Number(seconds || 0) / 60).toFixed(1)} phút` }

function App() {
  const [tab, setTab] = useState(() => {
    const requested = window.location.hash.slice(1)
    return ['studio', 'speech-to-speech', 'editor', 'voices', 'self-training', 'stt', 'training'].includes(requested) ? requested : 'studio'
  })
  const [voices, setVoices] = useState([])
  const [uploadedTraining, setUploadedTraining] = useState({})
  const [selected, setSelected] = useState('')
  const [voiceFile, setVoiceFile] = useState(null)
  const [voiceName, setVoiceName] = useState('')
  const [voiceLocale, setVoiceLocale] = useState('vi-VN')
  const [voiceSpeakerMode, setVoiceSpeakerMode] = useState('dominant')
  const [start, setStart] = useState(0)
  const [consent, setConsent] = useState(false)
  const [trainAfterUpload, setTrainAfterUpload] = useState(false)
  const [extraTrainingFiles, setExtraTrainingFiles] = useState({})
  const [extraTrainingConsent, setExtraTrainingConsent] = useState({})
  const [text, setText] = useState('Xin chào! Đây là giọng nói AI được tạo từ mẫu giọng của tôi.')
  const [deliveryMode, setDeliveryMode] = useState('natural')
  const [segmentSettings, setSegmentSettings] = useState({})
  const [video, setVideo] = useState(null)
  const [sourceMode, setSourceMode] = useState('mute')
  const [engine, setEngine] = useState('vieneu')
  const [ttsLocale, setTtsLocale] = useState('vi-VN')
  const [separateFile, setSeparateFile] = useState(null)
  const [separateResult, setSeparateResult] = useState(null)
  const [targetFile, setTargetFile] = useState(null)
  const [targetReference, setTargetReference] = useState(null)
  const [targetVoiceId, setTargetVoiceId] = useState('')
  const [targetMode, setTargetMode] = useState('keep')
  const [targetEnhance, setTargetEnhance] = useState(true)
  const [targetConsent, setTargetConsent] = useState(false)
  const [targetResult, setTargetResult] = useState(null)
  const [sttFile, setSttFile] = useState(null)
  const [language, setLanguage] = useState('auto')
  const [ttsResult, setTtsResult] = useState(null)
  const [editorIncoming, setEditorIncoming] = useState(null)
  const [sttResult, setSttResult] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sources, setSources] = useState([])
  const [benchmarks, setBenchmarks] = useState([])
  const [datasets, setDatasets] = useState([])
  const [jobs, setJobs] = useState([])
  const [runs, setRuns] = useState([])
  const [speakers, setSpeakers] = useState([])
  const [trainSource, setTrainSource] = useState('vivos')
  const [trainSpeaker, setTrainSpeaker] = useState('')
  const [trainDataset, setTrainDataset] = useState('')
  const [trainSteps, setTrainSteps] = useState(30)
  const [trainMaxClips, setTrainMaxClips] = useState(300)
  const [trainingQueue, setTrainingQueue] = useState({})
  const [guidedName, setGuidedName] = useState('Giọng của tôi')
  const [guidedLocale, setGuidedLocale] = useState('vi-VN')
  const [guidedConsent, setGuidedConsent] = useState(false)
  const [guidedSessions, setGuidedSessions] = useState([])
  const [guidedSession, setGuidedSession] = useState(null)
  const [guidedIndex, setGuidedIndex] = useState(0)
  const [guidedRecording, setGuidedRecording] = useState(false)
  const [guidedSeconds, setGuidedSeconds] = useState(0)
  const guidedRecorder = useRef(null)
  const guidedStream = useRef(null)
  const guidedClock = useRef(null)
  const guidedLimit = useRef(null)
  const { recording, toggle } = useRecorder(file => { setVoiceFile(file); setVoiceName(n => n || 'Giọng của tôi'); setVoiceSpeakerMode('manual') })
  const current = voices.find(v => v.id === selected)
  const compatibleVoices = voices.filter(v => !v.locale || v.locale === ttsLocale || (ttsLocale === 'en-US' && v.locale === 'en'))
  const guidedClip = guidedSession?.clips?.[String(guidedIndex)]
  const scriptLines = text.split('\n').map(line => line.trim()).filter(Boolean)
  function updateSegment(index, field, value) { setSegmentSettings(previous => ({ ...previous, [index]: { ...previous[index], [field]: value } })) }
  function splitSentences() {
    const split = text.replace(/([.!?。！？])\s+(?=\S)/g, '$1\n').replace(/\n{2,}/g, '\n')
    setText(split); setSegmentSettings({})
  }

  async function refresh() { const items = await api('/voices'); setVoices(items); setSelected(v => v || items[0]?.id || '') }
  useEffect(() => { refresh().catch(e => setError(e.message)) }, [])
  useEffect(() => { window.history.replaceState(null, '', `#${tab}`) }, [tab])
  useEffect(() => {
    let active = true
    async function refreshUploadedTraining() {
      const entries = await Promise.all(voices.filter(v => v.source_file).map(async v => {
        try { return [v.id, await api(`/voices/${v.id}/training`)] } catch { return [v.id, null] }
      }))
      if (active) {
        setUploadedTraining(Object.fromEntries(entries))
        const complete = entries.filter(([, state]) => state?.status === 'complete')
        if (complete.length) {
          refresh().catch(() => {})
        }
      }
    }
    refreshUploadedTraining()
    const timer = setInterval(refreshUploadedTraining, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [voices.map(v => v.id).join(',')])
  useEffect(() => { if (guidedSession?.status === 'complete') refresh().catch(e => setError(e.message)) }, [guidedSession?.status])
  useEffect(() => { setSelected(previous => {
    return compatibleVoices.some(v => v.id === previous) ? previous : compatibleVoices[0]?.id || ''
  }) }, [voices, ttsLocale, uploadedTraining])
  useEffect(() => () => { if (guidedClock.current) clearInterval(guidedClock.current); if (guidedLimit.current) clearTimeout(guidedLimit.current); guidedRecorder.current?.state === 'recording' && guidedRecorder.current.stop(); guidedStream.current?.getTracks().forEach(track => track.stop()) }, [])
  useEffect(() => {
    if (tab !== 'self-training') return
    let active = true
    const refreshGuided = () => api('/self-training/sessions').then(items => {
      if (!active) return
      setGuidedSessions(items)
      setGuidedSession(previous => items.find(item => item.id === previous?.id) || items[0] || null)
    }).catch(e => { if (active) setError(e.message) })
    refreshGuided()
    const timer = setInterval(refreshGuided, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [tab])
  async function refreshTraining() {
    const [s, d, j, r, b, q] = await Promise.all([api('/training/sources'), api('/training/datasets'), api('/training/jobs'), api('/training/runs'), api('/training/benchmarks'), api('/training/queue')])
    setSources(s); setDatasets(d); setJobs(j); setRuns(r); setBenchmarks(b); setTrainingQueue(q); setTrainDataset(v => v || d[0]?.id || '')
  }
  useEffect(() => {
    if (tab !== 'training') return
    refreshTraining().catch(e => setError(e.message))
    const timer = setInterval(() => refreshTraining().catch(() => {}), 4000)
    return () => clearInterval(timer)
  }, [tab])
  useEffect(() => {
    if (tab !== 'training') return
    api(`/training/speakers/${trainSource}`).then(v => { setSpeakers(v); setTrainSpeaker(v[0]?.id || '') }).catch(() => { setSpeakers([]); setTrainSpeaker('') })
  }, [tab, trainSource, sources.find(s => s.id === trainSource)?.downloaded])
  function resetMessages() { setError(''); setNotice('') }

  async function createVoice(e) {
    e.preventDefault(); resetMessages(); setBusy('voice')
    try {
      const form = new FormData(); form.append('file', voiceFile); form.append('name', voiceName); form.append('start', start); form.append('consent', consent); form.append('locale', voiceLocale); form.append('speaker_mode', voiceSpeakerMode)
      const voice = await api('/voices', { method: 'POST', body: form })
      await refresh(); setSelected(voice.id); setVoiceFile(null); setVoiceName(''); setStart(0); setConsent(false)
      if (trainAfterUpload) {
        try {
          const state = await api(`/voices/${voice.id}/train-full?clean=true`, { method: 'POST' })
          setUploadedTraining(previous => ({ ...previous, [voice.id]: state }))
          setNotice('Đã lưu toàn bộ file và bắt đầu chuẩn bị dữ liệu để fine-tune. Theo dõi tiến độ ở thẻ giọng.')
        } catch (trainError) {
          setError(`File đã lưu, nhưng chưa khởi động train: ${trainError.message}`)
        }
        setTab('voices')
      } else {
        setNotice('Đã lưu giọng tham chiếu. Bạn có thể tạo lời thoại ngay.'); setTab('studio')
      }
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function createTts(e) {
    e.preventDefault(); resetMessages(); setBusy('tts'); setTtsResult(null)
    try {
      const form = new FormData(); form.append('voice_id', selected); form.append('text', text)
      form.append('engine', engine); form.append('locale', ttsLocale)
      form.append('delivery', JSON.stringify({ mode: deliveryMode, segments: scriptLines.map((_, index) => segmentSettings[index] || {}) }))
      if (video) form.append('source', video)
      form.append('source_mode', sourceMode)
      setTtsResult(await api('/tts', { method: 'POST', body: form }))
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function separateAudio(e) {
    e.preventDefault(); resetMessages(); setBusy('separate'); setSeparateResult(null)
    try {
      const form = new FormData(); form.append('file', separateFile)
      setSeparateResult(await api('/separate', { method: 'POST', body: form }))
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function isolateTarget(e) {
    e.preventDefault(); resetMessages(); setBusy('target'); setTargetResult(null)
    try {
      const form = new FormData()
      form.append('file', targetFile)
      if (targetReference) form.append('reference', targetReference)
      else if (targetVoiceId) form.append('voice_id', targetVoiceId)
      else form.append('auto_dominant', 'true')
      form.append('locale', 'vi-VN')
      form.append('mode', targetMode)
      form.append('enhance', String(targetEnhance))
      form.append('consent', String(targetConsent))
      setTargetResult(await api('/target-voice', { method: 'POST', body: form }))
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function createStt(e) {
    e.preventDefault(); resetMessages(); setBusy('stt'); setSttResult(null)
    try {
      const form = new FormData(); form.append('file', sttFile); form.append('language', language)
      setSttResult(await api('/stt', { method: 'POST', body: form }))
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function removeVoice(id) {
    if (!window.confirm('Xóa giọng này khỏi thư viện?')) return
    resetMessages()
    try { await api(`/voices/${id}`, { method: 'DELETE' }); setSelected(v => v === id ? '' : v); await refresh() }
    catch (e) { setError(e.message) }
  }
  async function trainUploadedVoice(id) {
    resetMessages(); setBusy(`train-${id}`)
    try {
      const state = await api(`/voices/${id}/train-full?clean=true`, { method: 'POST' })
      setUploadedTraining(previous => ({ ...previous, [id]: state }))
      setNotice('Đã gửi tất cả bản thu của giọng này để lọc đúng người, phiên âm và kiểm tra trước fine-tune. Theo dõi tiến độ trên thẻ giọng.')
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function addTrainingFile(id) {
    resetMessages(); setBusy(`extra-${id}`)
    try {
      const form = new FormData()
      form.append('file', extraTrainingFiles[id])
      form.append('consent', String(extraTrainingConsent[id]))
      await api(`/voices/${id}/training-files`, { method: 'POST', body: form })
      setExtraTrainingFiles(previous => ({ ...previous, [id]: null }))
      setExtraTrainingConsent(previous => ({ ...previous, [id]: false }))
      await refresh()
      setNotice('Đã thêm bản thu mới. Bấm fine-tune lại sau khi đã nghe và kiểm tra mẫu giọng.')
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function sendTtsToEditor() {
    if (!ttsResult?.audio_url) return
    resetMessages()
    try {
      const response = await fetch(ttsResult.audio_url)
      if (!response.ok) throw new Error('Không tải được giọng AI vừa tạo.')
      const blob = await response.blob()
      setEditorIncoming({ id: crypto.randomUUID(), file: new File([blob], 'giong-ai-vocalab.wav', { type: 'audio/wav' }) })
      setTab('editor')
    } catch (error) { setError(error.message) }
  }
  async function prepareDataset(e) {
    e.preventDefault(); resetMessages(); setBusy('prepare')
    try {
      await api('/training/prepare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: trainSource, speaker: trainSpeaker, max_clips: Number(trainMaxClips) }) })
      setNotice('Đã xếp hàng chuẩn bị dữ liệu. Danh sách sẽ tự cập nhật.')
      await refreshTraining()
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function startTraining(e) {
    e.preventDefault(); resetMessages(); setBusy('train')
    try {
      await api('/training/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset: trainDataset, steps: Number(trainSteps) }) })
      setNotice('Đã xếp hàng fine-tune. Tiến độ sẽ tự cập nhật.')
      await refreshTraining()
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function evaluateRun(run) {
    resetMessages(); setBusy(`eval-${run}`)
    try {
      await api('/training/evaluate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run, count: 5 }) })
      setNotice('Đã xếp hàng đánh giá trên 5 câu test chưa dùng để train.')
      await refreshTraining()
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function startBenchmark() {
    resetMessages(); setBusy('benchmark')
    try {
      await api('/training/benchmark/fpt', { method: 'POST' })
      setNotice('Đã xếp hàng kiểm thử STT tiếng Việt trên FPT Open Speech.')
      await refreshTraining()
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }
  async function registerRun(run) {
    resetMessages(); setBusy(`register-${run}`)
    try {
      const voice = await api('/training/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run }) })
      await refresh(); setSelected(voice.id); setNotice('Đã thêm giọng fine-tune vào thư viện. Có thể dùng ngay trong Studio.')
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }

  async function startGuidedSession(e) {
    e.preventDefault(); resetMessages(); setBusy('guided-create')
    try {
      const session = await api('/self-training/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: guidedName, locale: guidedLocale, consent: guidedConsent }) })
      setGuidedSession(session); setGuidedSessions(previous => [session, ...previous]); setGuidedIndex(0)
      setNotice('Phiên thu âm đã sẵn sàng. Hãy đọc đúng từng câu hiển thị.')
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }

  async function toggleGuidedRecording() {
    if (guidedRecording) { guidedRecorder.current?.stop(); return }
    if (!guidedSession || guidedSession.status !== 'recording') return
    resetMessages()
    const sessionId = guidedSession.id
    const index = guidedIndex
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
    guidedStream.current = stream
    const chunks = []
    const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? { mimeType: 'audio/webm;codecs=opus' } : undefined
    const recorder = new MediaRecorder(stream, options)
    guidedRecorder.current = recorder
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data) }
    recorder.onstop = async () => {
      clearInterval(guidedClock.current); clearTimeout(guidedLimit.current)
      stream.getTracks().forEach(track => track.stop())
      setGuidedRecording(false); setBusy('guided-upload')
      try {
        const form = new FormData()
        form.append('file', new File(chunks, `cau-${index + 1}.webm`, { type: recorder.mimeType || 'audio/webm' }))
        const result = await api(`/self-training/sessions/${sessionId}/clips/${index}`, { method: 'POST', body: form })
        setGuidedSession(result.session)
        if (result.session.auto_train_error) setError(result.session.auto_train_error)
        if (result.clip.accepted) {
          const next = result.session.prompts.findIndex((_, i) => !result.session.clips[String(i)])
          if (next >= 0) setGuidedIndex(next)
          setNotice(result.session.status === 'recording' ? 'Đã lưu câu đọc rõ. Bạn có thể đọc câu tiếp theo.' : 'Đã đọc đủ 80 câu; web tự xếp hàng fine-tune và sẽ đánh giá giọng sau khi train.')
        } else setNotice('Máy nghe thấy câu khác với lời gợi ý. Hãy nghe lại, đọc lại hoặc tự xác nhận bản ghi.')
      } catch (e) { setError(e.message) } finally { setBusy('') }
    }
    recorder.start()
    setGuidedSeconds(0); setGuidedRecording(true)
    const started = Date.now()
    guidedClock.current = setInterval(() => setGuidedSeconds(Math.min(20, (Date.now() - started) / 1000)), 100)
    guidedLimit.current = setTimeout(() => { if (recorder.state === 'recording') recorder.stop() }, 20000)
  }

  async function acceptGuidedClip() {
    resetMessages(); setBusy('guided-accept')
    try {
      const result = await api(`/self-training/sessions/${guidedSession.id}/clips/${guidedIndex}/accept`, { method: 'POST' })
      setGuidedSession(result)
      if (result.auto_train_error) setError(result.auto_train_error)
      setNotice(result.status === 'recording' ? 'Đã xác nhận bản ghi theo đúng câu gợi ý.' : 'Đã đủ 80 câu; web tự xếp hàng fine-tune.')
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }

  async function trainGuidedVoice() {
    resetMessages(); setBusy('guided-train')
    try {
      const result = await api(`/self-training/sessions/${guidedSession.id}/train`, { method: 'POST' })
      setGuidedSession(result.session)
      setNotice('Đã xếp hàng fine-tune. Trang này sẽ tự cập nhật; giọng sẽ vào thư viện sau khi đánh giá xong.')
    } catch (e) { setError(e.message) } finally { setBusy('') }
  }

  const nav = [
    { id: 'studio', label: 'Studio giọng nói', icon: AudioLines },
    { id: 'speech-to-speech', label: 'Speech to speech', icon: Radio },
    { id: 'editor', label: 'Chỉnh sửa media', icon: Scissors },
    { id: 'stt', label: 'Chuyển lời nói', icon: FileText },
    { id: 'voices', label: 'Thư viện giọng', icon: FolderOpen },
    { id: 'self-training', label: 'Tự luyện giọng', icon: Mic2 },
    { id: 'training', label: 'Huấn luyện AI', icon: Cpu },
  ]
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><AudioLines size={25}/></span><span>voca<span className="brand-accent">lab</span><small>AI VOICE STUDIO</small></span></div>
      <div className="side-label">WORKSPACE</div>
      <nav>{nav.map(item => <button key={item.id} className={`nav-item ${tab === item.id ? 'selected' : ''}`} onClick={() => { setTab(item.id); resetMessages() }}><item.icon size={iconSize}/>{item.label}{item.id === 'voices' && <b>{voices.length}</b>}</button>)}</nav>
      <div className="sidebar-spacer"/>
      <div className="side-tip"><div className="side-tip-icon"><Sparkles size={17}/></div><strong>Mẹo thu âm</strong><p>Mẫu giọng rõ, ít tạp âm và chỉ có một người nói sẽ cho kết quả tốt hơn.</p></div>
      <div className="side-bottom"><span className="online-dot"/> Xử lý trên máy của bạn <span className="version">v1.0</span></div>
    </aside>

    <main className="main">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> {nav.find(n => n.id === tab)?.label}</div><div className="top-actions"><span className="local-badge"><span/> LOCAL STUDIO</span><div className="avatar">V</div></div></header>
      <div className="content">
        {error && <div className="alert error"><CircleHelp size={18}/><span>{error}</span><button onClick={() => setError('')}><X size={16}/></button></div>}
        {notice && <div className="alert success"><Sparkles size={18}/><span>{notice}</span><button onClick={() => setNotice('')}><X size={16}/></button></div>}

        {tab === 'studio' && <>
          <div className="page-heading"><div><div className="eyebrow"><span/> KHÔNG GIAN SÁNG TẠO ÂM THANH</div><h1>Biến ý tưởng thành <em>giọng nói.</em></h1><p>Tạo giọng AI từ video, podcast hoặc chính giọng nói của bạn. Viết lời thoại và để studio cất tiếng.</p></div><button className="button outline desktop-only" onClick={() => setTab('voices')}><Plus size={18}/> Tạo giọng mới</button></div>
          <div className="hero"><div className="hero-copy"><div className="hero-tag"><span className="hero-tag-dot"/> VOICE ENGINE · TIẾNG VIỆT</div><h2>Giọng của bạn.<br/><i>Bất kỳ câu chuyện nào.</i></h2><p>Chỉ cần vài giây giọng mẫu để bắt đầu sáng tạo.</p><button className="button white" onClick={() => setTab('voices')}><Plus size={18}/> Tạo giọng AI <span>↗</span></button></div><div className="hero-visual"><div className="orb orb-one"/><div className="orb orb-two"/><div className="hero-wave"><Waveform active/></div><div className="floating-card"><span className="floating-icon"><AudioLines size={19}/></span><span><strong>Studio sẵn sàng</strong><small>Thu âm · Tạo giọng · Lồng tiếng</small></span><span className="floating-live"/></div></div></div>
          <div className="section-heading"><div><span className="section-kicker">01 / TỔNG HỢP GIỌNG</span><h2>Tạo lời thoại</h2><p>Chọn giọng của bạn, nhập văn bản rồi tạo file âm thanh hoặc video lồng tiếng.</p></div></div>
          <div className="work-grid"><form className="panel tts-panel" onSubmit={createTts}><div className="panel-title"><span className="panel-icon purple"><Mic2 size={20}/></span><div><h3>Văn bản thành giọng nói</h3><p>Chọn một giọng đã lưu trong thư viện</p></div><span className="step-badge">BƯỚC 1</span></div>
            <label className="field-label" htmlFor="tts-locale">Chuẩn phát âm</label><div className="select-wrap"><select id="tts-locale" value={ttsLocale} onChange={e => { const value = e.target.value; setTtsLocale(value); const next = voices.find(v => v.locale === value || value === 'en-US' && v.locale === 'en'); setEngine(next?.kind === 'finetuned' ? 'vieneu' : value === 'en-US' ? 'voxcpm2' : 'vieneu') }}><option value="vi-VN">Tiếng Việt · luyện phát âm và thanh điệu</option><option value="en-US">English · General American</option></select><ChevronDown size={17}/></div><p className="fine-print">Giọng mẫu cần nói đúng ngôn ngữ và vùng giọng mong muốn; lựa chọn Anh Mỹ không tự sửa mọi giọng có accent khác.</p>
            <label className="field-label" htmlFor="voice-select">Giọng nói</label><div className="select-wrap"><select id="voice-select" value={selected} onChange={e => { setSelected(e.target.value); const voice = voices.find(v => v.id === e.target.value); setEngine(voice?.kind === 'finetuned' ? 'vieneu' : ttsLocale === 'en-US' ? 'voxcpm2' : 'vieneu') }} required><option value="">{compatibleVoices.length ? 'Chọn giọng nói' : 'Cần thu hoặc train giọng cho ngôn ngữ này'}</option>{compatibleVoices.map(v => <option value={v.id} key={v.id}>{v.name} · {v.locale || 'Chưa xác định'} · {v.kind === 'finetuned' ? 'đã train GPU' : 'mẫu tức thì'}</option>)}</select><ChevronDown size={17}/></div>
            <label className="field-label" htmlFor="engine-select">Mô hình tổng hợp</label><div className="select-wrap"><select id="engine-select" value={engine} onChange={e => setEngine(e.target.value)}><option value="vieneu" disabled={ttsLocale === 'en-US' && current?.kind !== 'finetuned'}>VieNeu-TTS v3 Turbo · giọng tham chiếu hoặc LoRA đã train</option><option value="voxcpm2">VoxCPM2 · Việt/Anh Mỹ, giọng tham chiếu</option></select><ChevronDown size={17}/></div>{engine === 'voxcpm2' && <p className="fine-print">VoxCPM2 dùng WAV tham chiếu; để dùng adapter đã train, chọn VieNeu. Giọng mẫu cùng ngôn ngữ giúp giữ phát âm chính xác hơn.</p>}
            {current && <div className="chosen-voice"><span className="chosen-avatar"><AudioLines size={18}/></span><span><strong>{current.name}</strong><small>Từ {current.source}</small></span><audio src={current.source_url || current.preview_url} controls preload="none" aria-label={current.source_url ? 'Nghe toàn bộ file gốc' : 'Nghe giọng mẫu'}/></div>}
            <div className="label-row"><label className="field-label" htmlFor="script">Nội dung lời thoại</label><span>{text.length}/5000</span></div><textarea id="script" value={text} maxLength={5000} onChange={e => setText(e.target.value)} placeholder="Nhập nội dung bạn muốn giọng AI đọc..." rows={6} required/>
            <div className="delivery-studio"><div className="delivery-heading"><div><strong>Đạo diễn giọng đọc</strong><small>Chọn phong cách và điều chỉnh từng dòng lời thoại</small></div><button type="button" onClick={splitSentences}>Tách theo câu</button></div>
              <div className="delivery-presets" role="group" aria-label="Phong cách giọng đọc">{[['natural','Tự nhiên'],['mc','MC'],['presenter','Thuyết trình'],['storyteller','Kể chuyện']].map(([id,label]) => <button type="button" key={id} className={deliveryMode === id ? 'active' : ''} onClick={() => setDeliveryMode(id)}>{label}</button>)}</div>
              <p className="delivery-hint">Mỗi dòng là một đoạn đọc riêng (tối đa 500 ký tự, 24 dòng). Cao độ và nhịp được xử lý sau tổng hợp; ngữ điệu còn phụ thuộc mẫu giọng gốc.</p>
              <div className="delivery-lines">{scriptLines.slice(0, 24).map((line, index) => <div className="delivery-line" key={`${index}-${line}`}><div className="delivery-line-title"><b>{String(index + 1).padStart(2, '0')}</b><span>{line}</span></div><div className="delivery-controls"><label>Cao độ <select aria-label={`Cao độ đoạn ${index + 1}`} value={segmentSettings[index]?.pitch ?? 0} onChange={e => updateSegment(index, 'pitch', Number(e.target.value))}><option value="-2">Trầm −2</option><option value="-1">Trầm −1</option><option value="0">Gốc</option><option value="1">Cao +1</option><option value="2">Cao +2</option></select></label><label>Nhịp <select aria-label={`Nhịp đoạn ${index + 1}`} value={segmentSettings[index]?.pace ?? 1} onChange={e => updateSegment(index, 'pace', Number(e.target.value))}><option value="0.85">Chậm</option><option value="1">Vừa</option><option value="1.12">Nhanh</option></select></label><label>Cuối câu <select aria-label={`Ngữ điệu đoạn ${index + 1}`} value={segmentSettings[index]?.contour ?? 'level'} onChange={e => updateSegment(index, 'contour', e.target.value)}><option value="level">Giữ đều</option><option value="rise">Lên giọng ↗</option><option value="fall">Xuống giọng ↘</option></select></label><label>Nhấn <select aria-label={`Nhấn đoạn ${index + 1}`} value={segmentSettings[index]?.gain ?? 0} onChange={e => updateSegment(index, 'gain', Number(e.target.value))}><option value="0">Bình thường</option><option value="2">Nhấn rõ</option><option value="3">Nhấn mạnh</option></select></label><label>Nghỉ <select aria-label={`Nghỉ sau đoạn ${index + 1}`} value={segmentSettings[index]?.pause ?? ''} onChange={e => updateSegment(index, 'pause', e.target.value === '' ? undefined : Number(e.target.value))}><option value="">Theo phong cách</option><option value="0">Không nghỉ</option><option value="250">0,25 giây</option><option value="500">0,5 giây</option><option value="1000">1 giây</option></select></label><label>Chèn tại giây <input aria-label={`Mốc chèn đoạn ${index + 1}`} type="number" min="0" max="7200" step="0.1" placeholder="Tự nối" value={segmentSettings[index]?.at ?? ''} onChange={e => updateSegment(index, 'at', e.target.value === '' ? undefined : Number(e.target.value))}/></label></div></div>)}</div>{scriptLines.length > 24 && <p className="delivery-error">Tối đa 24 dòng lời thoại. Hãy gộp bớt các dòng.</p>}{scriptLines.some(line => line.length > 500) && <p className="delivery-error">Mỗi dòng tối đa 500 ký tự. Hãy xuống dòng hoặc dùng “Tách theo câu”.</p>}
            </div>
            <div className="field-label video-label"><Video size={16}/> Audio hoặc video nguồn <span>(không bắt buộc)</span></div><FilePick file={video} onFile={setVideo} id="tts-video" label="Chọn audio hoặc video để chèn giọng" accept="audio/*,video/*,.mkv,.opus"/><SourcePreview file={video}/><p className="fine-print">Studio dùng toàn bộ file nguồn. Chỉ đoạn giọng tham chiếu của mô hình mới dài khoảng 8 giây.</p>
            {video && <><label className="field-label" htmlFor="source-mode">Xử lý âm thanh gốc</label><div className="select-wrap"><select id="source-mode" value={sourceMode} onChange={e => setSourceMode(e.target.value)}><option value="mute">Tắt toàn bộ âm thanh gốc</option><option value="keep">Giữ âm thanh gốc và chèn giọng AI</option><option value="duck">Hạ âm gốc khi AI nói</option><option value="remove_vocals">Tách và bỏ vocal gốc, giữ nhạc nền</option></select><ChevronDown size={17}/></div></>}
            <button className="button primary full" disabled={!!busy || !selected || !text.trim() || scriptLines.length > 24 || scriptLines.some(line => line.length > 500)}><WandSparkles size={18}/>{busy === 'tts' ? 'Đang tạo giọng nói…' : video ? 'Tạo âm thanh & video' : 'Tạo giọng nói'}<span className="button-arrow">→</span></button>
            <p className="fine-print">VieNeu giọng tham chiếu và giọng fine-tune chạy bằng GPU; lời thoại dài hoặc lúc GPU đang train có thể cần thêm thời gian.</p>{busy === 'tts' && <div className="tts-progress" role="status"><span className="record-dot"/> Đang chờ GPU nếu có lượt train, sau đó sẽ tổng hợp và xuất file…</div>}{error && <div className="tts-inline-error" role="alert">{error}</div>}
          </form>
          <div className="panel result-panel"><div className="panel-title"><span className="panel-icon blue"><Headphones size={20}/></span><div><h3>Nghe & xuất bản</h3><p>Kết quả tạo giọng của bạn xuất hiện tại đây</p></div></div>
            {ttsResult ? <div className="result-ready"><div className="result-art"><AudioLines size={42}/><Waveform active/></div><span className="result-label">TẠO THÀNH CÔNG · {ttsResult.locale}</span><h3>Lời thoại đã sẵn sàng</h3><audio className="result-audio" src={ttsResult.audio_url} controls/>{ttsResult.pronunciation_note && <p className="fine-print">{ttsResult.pronunciation_note}</p>}<button type="button" className="button outline send-editor" onClick={sendTtsToEditor}><Scissors size={16}/> Chèn giọng AI vào trang chỉnh sửa</button><div className="download-list"><a href={ttsResult.audio_url} download><FileAudio2 size={18}/> Tải âm thanh WAV <Download size={17}/></a>{ttsResult.video_url && <a href={ttsResult.video_url} download><Film size={18}/> Tải video MP4 <Download size={17}/></a>}</div>{ttsResult.video_url && <video className="video-preview" controls src={ttsResult.video_url}/>}</div> : <div className="empty-result"><div className="empty-orbit"><AudioLines size={42}/></div><h3>Chờ bản thu đầu tiên</h3><p>Nhập nội dung bên trái và chọn giọng nói để bắt đầu tạo âm thanh.</p><div className="empty-steps"><span><b>1</b> Chọn giọng</span><i/><span><b>2</b> Viết lời thoại</span><i/><span><b>3</b> Tạo âm thanh</span></div></div>}
          </div></div>
          <form className="panel separator-panel" onSubmit={isolateTarget}>
            <div className="panel-title"><span className="panel-icon blue"><Headphones size={20}/></span><div><h3>Giữ hoặc loại bỏ giọng của một người</h3><p>Tự tìm người nói nhiều nhất hoặc dùng mẫu giọng, rồi xử lý tối đa 60 phút.</p></div></div>
            <FilePick file={targetFile} onFile={setTargetFile} id="target-file" label="Chọn bản ghi hoặc video cần xử lý" accept="audio/*,video/*,.mkv,.opus"/>
            <label className="field-label" htmlFor="target-voice">Cách nhận diện người nói</label>
            <div className="select-wrap"><select id="target-voice" value={targetVoiceId} onChange={e => { setTargetVoiceId(e.target.value); if (e.target.value) setTargetReference(null) }} disabled={!!targetReference}><option value="">Tự tìm người có tổng thời lượng nói nhiều nhất</option>{voices.filter(v => v.kind !== 'finetuned').map(v => <option value={v.id} key={v.id}>{v.name} · {v.source}</option>)}</select><ChevronDown size={17}/></div>
            <FilePick file={targetReference} onFile={file => { setTargetReference(file); if (file) setTargetVoiceId('') }} id="target-reference" label="Hoặc tải mẫu giọng 3–8 giây của người đó" accept="audio/*,video/*,.mkv,.opus" hint="Đoạn một người nói, ít nhạc và tiếng ồn"/>
            <label className="field-label" htmlFor="target-mode">Video xuất ra</label><div className="select-wrap"><select id="target-mode" value={targetMode} onChange={e => setTargetMode(e.target.value)}><option value="keep">Chỉ giữ giọng người này</option><option value="remove">Bỏ giọng người này, giữ phần còn lại</option></select><ChevronDown size={17}/></div>
            <label className="consent"><input type="checkbox" checked={targetEnhance} onChange={e => setTargetEnhance(e.target.checked)}/><span>Giảm tiếng đường phố và tiếng ù trên giọng được giữ</span></label>
            <label className="consent"><input type="checkbox" checked={targetConsent} onChange={e => setTargetConsent(e.target.checked)}/><span>Tôi có quyền xử lý bản ghi và giọng nói này.</span></label>
            <button className="button primary" disabled={!!busy || !targetFile || !targetConsent}>{busy === 'target' ? 'Đang nhận diện và tách…' : 'Xử lý giọng người này'}</button>
            {targetResult && <><p className="fine-print">{targetResult.speaker_selection && <>Ước tính {targetResult.speaker_selection.speaker_clusters_estimate} cụm giọng; cụm chính chiếm {(targetResult.speaker_selection.dominant_speech_share * 100).toFixed(1)}% phần lời nói. </>}Đã giữ {targetResult.selected_chunks}/{targetResult.total_chunks} khung âm thanh. {targetResult.quality_note}</p><div className="download-list"><a href={targetResult.target_url} download>Chỉ giọng người này · WAV <Download size={17}/></a><a href={targetResult.without_target_url} download>Bỏ giọng người này · WAV <Download size={17}/></a>{targetResult.video_url && <a href={targetResult.video_url} download>Video đã xử lý · MP4 <Download size={17}/></a>}</div><audio src={targetMode === 'keep' ? targetResult.target_url : targetResult.without_target_url} controls/>{targetResult.video_url && <video className="video-preview" src={targetResult.video_url} controls/>}</>}
            <p className="fine-print">Mẫu giọng sạch quyết định độ chính xác. Hãy nghe kết quả trước khi dùng WAV đã làm sạch để fine-tune; đoạn nhiều người nói chồng nhau có thể còn sót.</p>
          </form>
          <form className="panel separator-panel" onSubmit={separateAudio}><div className="panel-title"><span className="panel-icon blue"><AudioLines size={20}/></span><div><h3>Tách giọng nói khỏi âm thanh gốc</h3><p>Demucs tách vocal và nền nhạc từ audio hoặc video. Hai stem có thể tải riêng.</p></div></div><FilePick file={separateFile} onFile={setSeparateFile} id="separate-file" label="Chọn audio hoặc video cần tách" accept="audio/*,video/*,.mkv,.opus"/><button className="button primary" disabled={!!busy || !separateFile}>{busy === 'separate' ? 'Đang tách vocal…' : 'Tách giọng / nhạc nền'}</button>{separateResult && <div className="download-list"><a href={separateResult.vocals_url} download>Tải vocal WAV <Download size={17}/></a><a href={separateResult.background_url} download>Tải nền WAV <Download size={17}/></a></div>}<p className="fine-print">Tách vocal khỏi nhạc nền; không phân biệt từng người khi nhiều người cùng nói.</p></form>
          <div className="feature-strip"><div><span className="feature-icon"><Radio size={20}/></span><strong>Giọng của riêng bạn</strong><small>Tạo từ bản ghi, podcast hoặc video</small></div><div><span className="feature-icon"><FileText size={20}/></span><strong>Phụ đề sẵn sàng</strong><small>Xuất văn bản và SRT có mốc thời gian</small></div><div><span className="feature-icon"><Film size={20}/></span><strong>Lồng tiếng video</strong><small>Gắn lời thoại AI vào video MP4</small></div></div>
        </>}

        {tab === 'speech-to-speech' && <SpeechToSpeech api={api} voices={voices} onError={setError} onNotice={setNotice}/>}

        <div style={{ display: tab === 'editor' ? 'block' : 'none' }}><Editor api={api} incoming={editorIncoming} onError={setError} onNotice={setNotice}/></div>

        {tab === 'voices' && <><div className="page-heading compact"><div><div className="eyebrow"><span/> THƯ VIỆN CỦA BẠN</div><h1>Mỗi giọng nói là <em>một câu chuyện.</em></h1><p>Tải video, podcast, file âm thanh hoặc thu trực tiếp bằng micro để tạo giọng tham chiếu.</p></div></div><div className="work-grid voice-grid"><form className="panel" onSubmit={createVoice}><div className="panel-title"><span className="panel-icon purple"><Plus size={20}/></span><div><h3>Tạo giọng AI mới</h3><p>Hỗ trợ bản ghi dài tối đa 60 phút</p></div></div><label className="field-label" htmlFor="voice-name">Tên giọng nói</label><input id="voice-name" className="text-input" placeholder="Ví dụ: Giọng của tôi" value={voiceName} onChange={e => setVoiceName(e.target.value)} maxLength={60} required/><label className="field-label" htmlFor="voice-locale">Ngôn ngữ của người nói trong mẫu</label><div className="select-wrap"><select id="voice-locale" value={voiceLocale} onChange={e => setVoiceLocale(e.target.value)}><option value="vi-VN">Tiếng Việt</option><option value="en-US">English · giọng Mỹ</option></select><ChevronDown size={17}/></div><label className="field-label" htmlFor="voice-speaker-mode">Cách chọn người nói</label><div className="select-wrap"><select id="voice-speaker-mode" value={voiceSpeakerMode} onChange={e => setVoiceSpeakerMode(e.target.value)}><option value="dominant">Tự tìm người có tổng thời lượng nói nhiều nhất</option><option value="manual">Tự chọn mốc có giọng sạch</option></select><ChevronDown size={17}/></div><label className="field-label">Nguồn giọng nói</label><FilePick file={voiceFile} onFile={setVoiceFile} id="voice-file" accept="audio/*,video/*,.mkv,.opus" label="Chọn audio, podcast hoặc video" hint="Tối đa 60 phút · file gốc được giữ đầy đủ"/><div className="or-divider"><span>HOẶC</span></div><button type="button" className={`record-button ${recording ? 'recording' : ''}`} onClick={() => toggle().catch(e => setError(e.message))}>{recording ? <Pause size={19}/> : <Mic2 size={19}/>} {recording ? 'Dừng và dùng bản ghi' : 'Thu giọng nói của tôi'}{recording && <span className="record-dot"/>}</button>{voiceSpeakerMode === 'manual' ? <><label className="field-label" htmlFor="start">Bắt đầu lấy mẫu tại giây thứ</label><div className="start-row"><input id="start" className="text-input" type="number" min="0" step="0.1" value={start} onChange={e => setStart(e.target.value)}/><span>Chỉ lấy 8 giây làm mẫu giọng AI; file gốc vẫn được lưu đầy đủ.</span></div></> : <p className="fine-print">Hệ thống phân cụm các đoạn lời nói, chọn cụm có tổng thời lượng lớn nhất và lấy một đoạn rõ 3–8 giây làm mẫu.</p>}<label className="consent"><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)}/><span>Tôi sở hữu giọng nói này hoặc đã được người nói cho phép sử dụng.</span></label><label className="consent"><input type="checkbox" checked={trainAfterUpload} onChange={e => setTrainAfterUpload(e.target.checked)}/><span>Lọc đúng người, giảm nhiễu rồi fine-tune nếu còn ít nhất 5 phút và 50 đoạn lời nói.</span></label><button className="button primary full" disabled={!!busy || !voiceFile || !consent}><Sparkles size={18}/>{busy === 'voice' ? (voiceSpeakerMode === 'dominant' ? 'Đang tìm người nói chính…' : 'Đang xử lý file…') : trainAfterUpload ? 'Lưu & train toàn bộ file' : 'Chỉ lưu mẫu tham chiếu'}<span className="button-arrow">→</span></button><p className="fine-print">Giọng được tạo tức thì từ đoạn tham chiếu; fine-tune dùng toàn bộ các đoạn đạt chuẩn của đúng cụm giọng.</p></form><div className="panel library-panel"><div className="panel-title"><span className="panel-icon blue"><FolderOpen size={20}/></span><div><h3>Giọng đã lưu</h3><p>{voices.length} giọng nói trong thư viện</p></div></div>{voices.length ? <div className="voice-list">{voices.map(v => <div className="voice-card" key={v.id}><div className="voice-card-top"><span className="voice-card-icon"><AudioLines size={22}/></span><div><strong>{v.name}</strong><small>{v.source} · {v.locale || 'Chưa xác định ngôn ngữ'} · {v.kind === 'finetuned' ? 'đã train GPU' : 'mẫu tức thì'}</small></div><button title="Xóa giọng" onClick={() => removeVoice(v.id)}><Trash2 size={17}/></button></div><Waveform/><div className="voice-card-bottom"><span><Clock3 size={14}/>{v.source_url ? `File gốc ${minutes(v.duration)} · mẫu ngắn chỉ dùng cho chế độ tham chiếu` : `${v.duration}s mẫu`}</span>{!v.source_url && <audio src={v.preview_url} controls preload="none" aria-label={`Nghe ${v.name}`}/>}</div>{v.source_url && <div className="full-source-controls"><strong>Nghe toàn bộ file gốc · {minutes(v.duration)}</strong><audio src={v.source_url} controls preload="none" aria-label={`Nghe toàn bộ file ${v.name}`}/><a className="source-download" href={v.source_url} download>Tải file gốc đầy đủ <Download size={14}/></a></div>}{v.speaker_selection && <p className="fine-print">Tự chọn cụm chính: {(v.speaker_selection.dominant_speech_share * 100).toFixed(1)}% phần lời nói · ước tính {v.speaker_selection.speaker_clusters_estimate} cụm giọng.</p>}{v.source_file && <div className="full-train-state"><p><strong>Mẫu giọng sẽ dùng để nhận diện:</strong></p><audio src={v.preview_url} controls preload="none" aria-label={`Mẫu giọng ${v.name}`}/><FilePick file={extraTrainingFiles[v.id] || null} onFile={file => setExtraTrainingFiles(previous => ({ ...previous, [v.id]: file }))} id={`extra-training-${v.id}`} label="Thêm bản thu sạch cùng người nói" accept="audio/*,video/*,.mkv,.opus" hint="3 giây đến 60 phút · bản thu mới, đúng người"/>{v.training_files?.length > 0 && <p>Đã thêm {v.training_files.length} bản thu · {minutes(v.training_files.reduce((sum, row) => sum + row.duration, 0))}</p>}{extraTrainingFiles[v.id] && <><label className="consent"><input type="checkbox" checked={!!extraTrainingConsent[v.id]} onChange={e => setExtraTrainingConsent(previous => ({ ...previous, [v.id]: e.target.checked }))}/><span>Tôi có quyền dùng bản thu mới của đúng người này để huấn luyện.</span></label><button type="button" className="button outline" disabled={!!busy || !extraTrainingConsent[v.id]} onClick={() => addTrainingFile(v.id)}>Thêm bản thu</button></>}{!uploadedTraining[v.id] || uploadedTraining[v.id]?.status === 'not_started' ? <button type="button" className="button primary full" disabled={!!busy} onClick={() => trainUploadedVoice(v.id)}><Cpu size={16}/> Lọc giọng và fine-tune file này</button> : uploadedTraining[v.id]?.status === 'failed' ? <p className="train-error">Train lỗi: {uploadedTraining[v.id].error}</p> : <><strong>Fine-tune: {({ queued:'Đang xếp hàng', preparing:'Chuẩn bị dữ liệu', waiting_for_gpu:'Chờ GPU', training:'GPU đang huấn luyện', evaluating:'Đang kiểm tra', complete:'Hoàn tất', needs_more_data:'Cần thêm giọng sạch', needs_review:'Cần kiểm tra bản train thử' })[uploadedTraining[v.id].status] || uploadedTraining[v.id].status}</strong><p>File gốc {minutes(uploadedTraining[v.id].source_seconds || v.duration)} · lời nói đạt chuẩn {uploadedTraining[v.id].usable_seconds == null ? 'đang đo' : minutes(uploadedTraining[v.id].usable_seconds)} · {uploadedTraining[v.id].train_clips || 0} đoạn dùng để train toàn bộ file</p>{uploadedTraining[v.id].reason && <p className="train-error">{uploadedTraining[v.id].reason}</p>}{uploadedTraining[v.id].steps && <p>{uploadedTraining[v.id].optimizer_updates || 0}/{uploadedTraining[v.id].steps} bước tối ưu dự kiến · batch {uploadedTraining[v.id].batch_size || 'đang chọn'}</p>}{['complete','needs_more_data','needs_review'].includes(uploadedTraining[v.id].status) && uploadedTraining[v.id].trained_voice_id && <button type="button" className="use-voice" onClick={() => { setSelected(uploadedTraining[v.id].trained_voice_id); setTtsLocale(v.locale); setEngine('vieneu'); setTab('studio') }}>Dùng giọng đã train trong Studio →</button>}</>}{['complete','needs_more_data','needs_review','failed'].includes(uploadedTraining[v.id]?.status) && <button type="button" className="button primary full" disabled={!!busy} onClick={() => trainUploadedVoice(v.id)}><Cpu size={16}/> Lọc và fine-tune lại từ tất cả bản thu</button>}</div>}<button className="use-voice" onClick={() => { setSelected(v.id); setTtsLocale(v.locale === 'en' ? 'en-US' : v.locale || 'vi-VN'); setEngine(v.kind === 'finetuned' ? 'vieneu' : v.locale === 'en' || v.locale === 'en-US' ? 'voxcpm2' : 'vieneu'); setTab('studio') }}>Dùng giọng này <span>→</span></button></div>)}</div> : <div className="empty-result small"><div className="empty-orbit"><Mic2 size={38}/></div><h3>Chưa có giọng nào</h3><p>Tạo giọng đầu tiên bằng cách tải lên một đoạn nói ngắn.</p></div>}</div></div></>}

        {tab === 'self-training' && <>
          <div className="page-heading compact"><div><div className="eyebrow"><span/> THU ÂM · FINE-TUNE · ĐÁNH GIÁ</div><h1>Dạy AI nói bằng <em>giọng của bạn.</em></h1><p>Chọn tiếng Việt hoặc Anh Mỹ, đọc theo từng câu, nghe lại bản ghi và để web tự huấn luyện khi đủ mẫu.</p></div></div>
          <div className="work-grid guided-grid"><form className="panel" onSubmit={startGuidedSession}><div className="panel-title"><span className="panel-icon purple"><Mic2 size={20}/></span><div><h3>Tạo phiên đọc mới</h3><p>80 câu hướng dẫn cho từng ngôn ngữ</p></div></div><label className="field-label" htmlFor="guided-name">Tên giọng của bạn</label><input id="guided-name" className="text-input" value={guidedName} onChange={e => setGuidedName(e.target.value)} maxLength={60} required/><label className="field-label" htmlFor="guided-locale">Ngôn ngữ và chuẩn phát âm</label><div className="select-wrap"><select id="guided-locale" value={guidedLocale} onChange={e => setGuidedLocale(e.target.value)}><option value="vi-VN">Tiếng Việt · luyện dấu và thanh điệu</option><option value="en-US">English · General American</option></select><ChevronDown size={17}/></div><label className="consent"><input type="checkbox" checked={guidedConsent} onChange={e => setGuidedConsent(e.target.checked)}/><span>Đây là giọng của tôi hoặc tôi được người nói cho phép ghi âm và huấn luyện.</span></label><button className="button primary full" disabled={!!busy || !guidedConsent}><Plus size={18}/> Tạo phiên đọc</button><p className="fine-print">Web tự xếp hàng train khi đủ 80 câu và 5 phút bản ghi hợp lệ. Có thể bấm train sớm từ 50 câu và 5 phút; khoảng 10 phút âm thanh sạch thường giúp giữ giọng tốt hơn.</p></form><div className="panel guided-list"><div className="panel-title"><span className="panel-icon blue"><FolderOpen size={20}/></span><div><h3>Phiên đang có</h3><p>Có thể tiếp tục sau khi tải lại trang</p></div></div>{guidedSessions.length ? guidedSessions.map(item => <button type="button" key={item.id} className={`guided-session-row ${guidedSession?.id === item.id ? 'selected' : ''}`} onClick={() => { setGuidedSession(item); setGuidedIndex(Math.max(0, item.prompts.findIndex((_, i) => !item.clips[String(i)]))) }}><strong>{item.name}</strong><span>{item.locale} · {item.accepted_count}/80 câu · {Math.floor(item.accepted_seconds / 60)} phút · {item.status}</span></button>) : <p className="fine-print">Chưa có phiên thu âm. Hãy tạo phiên ở bên trái.</p>}</div></div>
          {guidedSession && <div className="panel guided-workspace"><div className="panel-title"><span className="panel-icon purple"><AudioLines size={20}/></span><div><h3>{guidedSession.name} · {guidedSession.locale}</h3><p>{guidedSession.accepted_count}/80 câu đã chấp nhận · {Math.floor(guidedSession.accepted_seconds / 60)}:{String(Math.floor(guidedSession.accepted_seconds % 60)).padStart(2, '0')} phút hợp lệ · trạng thái: {guidedSession.status}</p></div></div><div className="guided-progress"><span style={{ width: `${Math.min(100, guidedSession.accepted_seconds / 300 * 100)}%` }}/></div><p className="fine-print">Đọc trong phòng yên tĩnh, giữ micro cách miệng đều. Mỗi câu từ 3 đến 15 giây theo recipe VieNeu mới. Thu lại câu có tiếng ồn hoặc phiên âm sai.</p>{guidedSession.status === 'recording' && <><div className="guided-prompt-head"><span>CÂU {guidedIndex + 1} / {guidedSession.prompts.length}</span><span>{guidedClip ? guidedClip.accepted ? 'Đã chấp nhận' : 'Cần kiểm tra' : 'Chưa thu'}</span></div><blockquote className="guided-prompt">{guidedSession.prompts[guidedIndex]}</blockquote><div className="guided-record-actions"><button type="button" className={`record-button ${guidedRecording ? 'recording' : ''}`} disabled={!!busy} onClick={() => toggleGuidedRecording().catch(e => setError(e.message))}>{guidedRecording ? <Pause size={19}/> : <Mic2 size={19}/>} {guidedRecording ? `Dừng ghi · ${guidedSeconds.toFixed(1)}s` : guidedClip ? 'Thu lại câu này' : 'Bật micro và đọc câu này'}</button>{busy === 'guided-upload' && <span>Đang kiểm tra giọng đọc…</span>}</div>{guidedClip && <div className="guided-feedback"><audio controls src={`/api/self-training/sessions/${guidedSession.id}/clips/${guidedIndex}/audio`}/><p><strong>Máy nghe:</strong> {guidedClip.recognized || 'Không nhận rõ lời nói'}</p><p>Khớp lời gợi ý: {Math.round(guidedClip.match_score * 100)}% · {guidedClip.duration}s · {guidedClip.accepted ? 'Đã lưu để train' : 'Hãy đọc lại hoặc xác nhận nếu máy nghe sai'}</p>{guidedClip.needs_review && <button type="button" className="button outline" disabled={!!busy} onClick={acceptGuidedClip}>Tôi đã nghe và xác nhận câu này</button>}</div>}<div className="guided-prompt-list">{guidedSession.prompts.map((_, index) => <button type="button" key={index} title={`Câu ${index + 1}`} className={`${index === guidedIndex ? 'active' : ''} ${guidedSession.clips[String(index)]?.accepted ? 'done' : guidedSession.clips[String(index)] ? 'review' : ''}`} onClick={() => setGuidedIndex(index)} disabled={guidedRecording || !!busy}>{index + 1}</button>)}</div><button type="button" className="button primary" disabled={!!busy || !guidedSession.ready_to_train} onClick={trainGuidedVoice}><Cpu size={18}/> {busy === 'guided-train' ? 'Đang xếp hàng…' : 'Đủ mẫu · tự fine-tune giọng này'}</button>{!guidedSession.ready_to_train && <p className="fine-print">Còn thiếu {Math.max(0, guidedSession.min_clips - guidedSession.accepted_count)} câu và {Math.max(0, Math.ceil(guidedSession.min_seconds - guidedSession.accepted_seconds))} giây trước khi có thể train.</p>}</>}{guidedSession.status !== 'recording' && <div className="guided-status"><Cpu size={30}/><h3>{guidedSession.status === 'complete' ? 'Giọng đã huấn luyện và thêm vào thư viện' : guidedSession.status === 'failed' ? 'Huấn luyện gặp lỗi' : guidedSession.status === 'waiting_for_gpu' ? 'Đang chờ GPU rảnh' : guidedSession.status === 'evaluating' ? 'Đang đánh giá trên câu chưa dùng để train' : 'Đang chuẩn bị hoặc huấn luyện'}</h3>{guidedSession.error && <p>{guidedSession.error}</p>}{guidedSession.dispatch_error && <><p>{guidedSession.dispatch_error}</p><button type="button" className="button outline" disabled={!!busy} onClick={trainGuidedVoice}>Thử khởi động lại worker →</button></>}{guidedSession.status === 'complete' && <><p>{guidedSession.optimizer_updates} bước tối ưu · WER giọng AI: {Math.round(guidedSession.generated_wer * 100)}% · {guidedSession.test_items} câu test</p><button type="button" className="button primary" onClick={() => { setTtsLocale(guidedSession.locale); setEngine('vieneu'); setSelected(guidedSession.voice_id); setTab('studio') }}>Dùng giọng đã train trong Studio →</button></>}</div>}</div>}
        </>}

        {tab === 'stt' && <><div className="page-heading compact"><div><div className="eyebrow"><span/> CHUYỂN LỜI NÓI THÀNH VĂN BẢN</div><h1>Nghe thật rõ. <em>Ghi thật chuẩn.</em></h1><p>Biến video, podcast và bản ghi âm thành văn bản có mốc thời gian, sẵn sàng làm phụ đề.</p></div></div><div className="work-grid"><form className="panel" onSubmit={createStt}><div className="panel-title"><span className="panel-icon purple"><UploadCloud size={20}/></span><div><h3>Tải tệp cần chuyển</h3><p>Hỗ trợ cả audio và video</p></div></div><FilePick file={sttFile} onFile={setSttFile} id="stt-file" accept="audio/*,video/*,.mkv,.opus" label="Chọn video, podcast hoặc audio"/><label className="field-label" htmlFor="language">Ngôn ngữ trong tệp</label><div className="select-wrap"><select id="language" value={language} onChange={e => setLanguage(e.target.value)}><option value="auto">Tự nhận diện</option><option value="vi">Tiếng Việt</option><option value="en">English</option></select><ChevronDown size={17}/></div><div className="stt-help"><FileText size={20}/><p>Kết quả bao gồm văn bản thuần và phụ đề SRT với mốc thời gian theo từng câu.</p></div><button className="button primary full" disabled={!!busy || !sttFile}><FileText size={18}/>{busy === 'stt' ? 'Đang nghe và phiên âm…' : 'Chuyển thành văn bản'}<span className="button-arrow">→</span></button><p className="fine-print">Lần đầu sử dụng có thể mất thêm thời gian để tải mô hình STT.</p></form><div className="panel transcript-panel"><div className="panel-title"><span className="panel-icon blue"><FileText size={20}/></span><div><h3>Bản phiên âm</h3><p>Nội dung có thể sao chép và tải xuống</p></div></div>{sttResult ? <div className="transcript-content"><div className="transcript-toolbar"><span><span className="online-dot"/> {sttResult.language?.toUpperCase()} · {sttResult.segments.length} đoạn</span><button onClick={() => navigator.clipboard.writeText(sttResult.text).then(() => setNotice('Đã sao chép bản phiên âm.'))}>Sao chép</button></div><div className="segments">{sttResult.segments.length ? sttResult.segments.map((s, i) => <div className="segment" key={i}><time>{Math.floor(s.start / 60).toString().padStart(2,'0')}:{Math.floor(s.start % 60).toString().padStart(2,'0')}</time><p>{s.text}</p></div>) : <p>Không tìm thấy lời nói trong tệp.</p>}</div><div className="download-list"><a href={sttResult.txt_url} download><FileText size={18}/> Tải văn bản TXT <Download size={17}/></a><a href={sttResult.srt_url} download><Film size={18}/> Tải phụ đề SRT <Download size={17}/></a></div></div> : <div className="empty-result"><div className="empty-orbit"><FileText size={40}/></div><h3>Lời nói thành văn bản</h3><p>Tải một tệp lên và bản phiên âm sẽ hiện ở đây cùng mốc thời gian.</p></div>}</div></div></>}
        {tab === 'training' && <><div className="page-heading compact"><div><div className="eyebrow"><span/> DỮ LIỆU · FINE-TUNE · ĐÁNH GIÁ</div><h1>Dạy AI một <em>giọng riêng.</em></h1><p>Mỗi lần fine-tune chỉ học một người nói. Mẫu được chia riêng train, validation và test để đo chất lượng trung thực.</p></div></div>
          <div className="training-sources">{sources.map(s => <div className="source-card" key={s.id}><span className="source-icon"><Database size={20}/></span><div><strong>{s.name}</strong><small>{s.language.toUpperCase()} · {s.license}</small>{!s.trainable && <small>Benchmark STT · không có ID người nói để fine-tune TTS</small>}{!s.trainable && benchmarks.find(b => b.source === s.id) && <small>WER: {(benchmarks.find(b => b.source === s.id).wer * 100).toFixed(1)}% / {benchmarks.find(b => b.source === s.id).items} câu</small>}{!s.trainable && s.downloaded && <button className="source-action" onClick={startBenchmark} disabled={!!busy}>Chạy kiểm thử STT →</button>}</div><span className={`source-state ${s.downloaded ? 'ready' : ''}`}>{s.downloaded ? 'Đã tải' : 'Chưa tải'}</span></div>)}</div>
          <div className="work-grid"><form className="panel" onSubmit={prepareDataset}><div className="panel-title"><span className="panel-icon purple"><Database size={20}/></span><div><h3>Chuẩn bị dữ liệu</h3><p>Chọn một nguồn và một người nói</p></div><span className="step-badge">BƯỚC 1</span></div><label className="field-label" htmlFor="train-source">Nguồn công khai</label><div className="select-wrap"><select id="train-source" value={trainSource} onChange={e => { setTrainSource(e.target.value); setTrainMaxClips(e.target.value === 'ljspeech' ? 5000 : 300) }}>{sources.filter(s => s.trainable).map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select><ChevronDown size={17}/></div><label className="field-label" htmlFor="train-speaker">Người nói</label><div className="select-wrap"><select id="train-speaker" value={trainSpeaker} onChange={e => setTrainSpeaker(e.target.value)} disabled={!speakers.length}>{speakers.length ? speakers.map(s => <option key={s.id} value={s.id}>{s.id} · {s.clips} mẫu</option>) : <option value="">Chưa có archive trên máy</option>}</select><ChevronDown size={17}/></div><label className="field-label" htmlFor="train-max-clips">Số mẫu tối đa</label><input id="train-max-clips" className="text-input" type="number" min="30" max="10000" value={trainMaxClips} onChange={e => setTrainMaxClips(e.target.value)}/><p className="fine-print">Âm thanh được chuẩn hóa 24 kHz, loại mẫu im lặng, câu trùng và tách tập theo câu. Quyền sử dụng phụ thuộc giấy phép gốc.</p><button className="button primary full" disabled={!!busy || !trainSpeaker}><Database size={17}/>{busy === 'prepare' ? 'Đang xếp hàng…' : `Chuẩn bị ${trainMaxClips} mẫu`}<span className="button-arrow">→</span></button></form>
          <form className="panel" onSubmit={startTraining}><div className="panel-title"><span className="panel-icon blue"><Cpu size={20}/></span><div><h3>Fine-tune LoRA</h3><p>VieNeu-TTS v3 Turbo · cần GPU CUDA</p></div><span className="step-badge">BƯỚC 2</span></div><label className="field-label" htmlFor="train-dataset">Bộ dữ liệu đã chuẩn bị</label><div className="select-wrap"><select id="train-dataset" value={trainDataset} onChange={e => setTrainDataset(e.target.value)} disabled={!datasets.length}>{datasets.length ? datasets.map(d => <option key={d.id} value={d.id}>{d.id} · {d.counts.train} train / {d.counts.test} test</option>) : <option value="">Chưa có bộ dữ liệu</option>}</select><ChevronDown size={17}/></div><label className="field-label" htmlFor="train-steps">Số bước cập nhật</label><input id="train-steps" className="text-input" type="number" min="1" max="10000" value={trainSteps} onChange={e => setTrainSteps(e.target.value)}/><p className="fine-print">Adapter và log được lưu cùng mã SHA-256. Kết quả cần đánh giá trên tập test trước khi sử dụng.</p><button className="button primary full" disabled={!!busy || !trainDataset}><Cpu size={17}/>{busy === 'train' ? 'Đang xếp hàng…' : 'Bắt đầu fine-tune'}<span className="button-arrow">→</span></button></form></div>
          <div className="panel training-history"><div className="panel-title"><span className="panel-icon purple"><Clock3 size={20}/></span><div><h3>Tiến độ & kết quả</h3><p>Tự cập nhật mỗi 4 giây</p></div></div>{Object.entries(trainingQueue).map(([name, queue]) => <div className="history-row" key={name}><span>Hàng đợi tự động</span><strong>{name === 'extended_train' ? 'Huấn luyện mở rộng' : 'Huấn luyện ban đầu'}</strong><em className={queue.status}>{queue.status}</em><small>{Object.values(queue.runs || {}).filter(r => r.status === 'complete').length}/{queue.plan?.length || 0} giọng hoàn tất</small></div>)}{jobs.length === 0 && runs.length === 0 ? <p className="history-empty">Chưa có phiên chuẩn bị hoặc huấn luyện.</p> : <div className="history-list">{jobs.slice(0, 6).map(j => <div className="history-row" key={j.id}><span>{j.kind === 'train' ? 'Fine-tune' : j.kind === 'evaluate' ? 'Đánh giá' : 'Chuẩn bị dữ liệu'}</span><strong>{j.result?.dataset || j.result?.run || j.id.slice(0, 8)}</strong><em className={j.status}>{j.status === 'complete' ? 'Hoàn tất' : j.status === 'failed' ? 'Thất bại' : j.status === 'running' ? 'Đang chạy' : 'Đang chờ'}</em>{j.error && <small>{j.error}</small>}</div>)}{runs.slice(0, 5).map(r => <div className="history-row" key={r.run}><span>Adapter</span><strong>{r.run}</strong><em className={r.status}>{r.status === 'evaluated' ? 'Đã đánh giá' : r.status === 'trained_pending_evaluation' ? 'Chờ đánh giá' : r.status === 'failed' ? 'Thất bại' : 'Đang chạy'} · {r.updates} bước</em>{r.adapter_sha256 && <small>SHA-256: {r.adapter_sha256.slice(0, 24)}…</small>}{r.generated_wer != null && <small>WER WAV gốc: {r.reference_wer != null ? (r.reference_wer * 100).toFixed(1) : "?"}% · giọng AI: {(r.generated_wer * 100).toFixed(1)}% trên {r.test_items} câu test</small>}{r.speaker_similarity && <small>Cosine giọng so với mẫu: WAV gốc {r.speaker_similarity.original_to_enrollment.toFixed(3)} · AI {r.speaker_similarity.generated_to_enrollment.toFixed(3)}</small>}{r.status === 'trained_pending_evaluation' && <button className="history-action" disabled={!!busy} onClick={() => evaluateRun(r.run)}>Đánh giá test →</button>}{r.status === 'evaluated' && (r.registered ? <span className="registered-label">Đã trong thư viện</span> : <button className="history-action" disabled={!!busy} onClick={() => registerRun(r.run)}>Thêm vào thư viện →</button>)}</div>)}</div>}</div>
        </>}
      </div>
    </main>
  </div>
}

createRoot(document.getElementById('root')).render(<App/>)

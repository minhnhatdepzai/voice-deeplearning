import React, { useEffect, useRef, useState } from 'react'
import { AudioLines, ChevronLeft, ChevronRight, Download, Film, Music2, Pause, Play, Plus, Scissors, Trash2, UploadCloud } from 'lucide-react'
import './editor.css'

const uid = () => crypto.randomUUID()
const videoFile = file => file.type.startsWith('video/') || (!file.type && /\.(mp4|mov|mkv|webm)$/i.test(file.name))
const stamp = seconds => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`

export default function Editor({ api, incoming, onError, onNotice }) {
  const [assets, setAssets] = useState([])
  const [clips, setClips] = useState([])
  const [selected, setSelected] = useState('')
  const [playhead, setPlayhead] = useState(0)
  const [overlay, setOverlay] = useState(null)
  const [overlayAt, setOverlayAt] = useState(0)
  const [overlayVolume, setOverlayVolume] = useState(1)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const mediaRef = useRef(null)
  const assetsRef = useRef([])
  const overlayRef = useRef(null)
  const picked = clips.find(clip => clip.id === selected) || clips[0]
  const activeAsset = assets.find(asset => asset.id === picked?.sourceId)
  const total = clips.reduce((sum, clip) => sum + Math.max(0, (clip.end ?? clip.start) - clip.start), 0)

  useEffect(() => { assetsRef.current = assets }, [assets])
  useEffect(() => { overlayRef.current = overlay }, [overlay])
  useEffect(() => () => { assetsRef.current.forEach(asset => URL.revokeObjectURL(asset.url)); if (overlayRef.current) URL.revokeObjectURL(overlayRef.current.url) }, [])
  useEffect(() => {
    if (!incoming?.file) return
    setOverlay(previous => { if (previous) URL.revokeObjectURL(previous.url); return { file: incoming.file, url: URL.createObjectURL(incoming.file) } })
    onNotice('Đã đưa giọng AI vào lớp âm thanh của trình chỉnh sửa.')
  }, [incoming?.id])
  useEffect(() => { if (picked) setPlayhead(picked.start) }, [selected, picked?.sourceId])
  useEffect(() => {
    if (!mediaRef.current || !picked) return
    mediaRef.current.currentTime = picked.start
    mediaRef.current.volume = picked.muted ? 0 : Math.min(1, picked.volume)
  }, [selected, activeAsset?.id, picked?.start, picked?.volume, picked?.muted])

  function addFiles(fileList) {
    const files = Array.from(fileList || [])
    if (!files.length) return
    if (assets.length + files.length > 12) { onError('Mỗi dự án dùng tối đa 12 tệp nguồn.'); return }
    if (files.some(file => file.size > 500 * 1024 * 1024)) { onError('Mỗi tệp tối đa 500 MB.'); return }
    const added = files.map(file => ({ id: uid(), file, url: URL.createObjectURL(file), duration: null, video: videoFile(file) }))
    const addedClips = added.map(asset => ({ id: uid(), sourceId: asset.id, start: 0, end: null, volume: 1, muted: false }))
    setAssets(previous => [...previous, ...added])
    setClips(previous => [...previous, ...addedClips])
    if (!selected) setSelected(addedClips[0].id)
    setResult(null)
  }

  function metadata(assetId, duration) {
    if (!Number.isFinite(duration) || duration <= 0) return
    setAssets(previous => previous.map(asset => asset.id === assetId ? { ...asset, duration } : asset))
    setClips(previous => previous.map(clip => clip.sourceId === assetId && clip.end == null ? { ...clip, end: duration } : clip))
  }

  function removeAsset(id) {
    const asset = assets.find(item => item.id === id)
    if (asset) URL.revokeObjectURL(asset.url)
    setAssets(previous => previous.filter(item => item.id !== id))
    setClips(previous => previous.filter(clip => clip.sourceId !== id))
    if (picked?.sourceId === id) setSelected('')
    setResult(null)
  }

  function updateClip(id, update) {
    setClips(previous => previous.map(clip => clip.id === id ? { ...clip, ...update } : clip))
    setResult(null)
  }

  function split() {
    if (!picked || picked.end == null || playhead <= picked.start + .1 || playhead >= picked.end - .1) return
    const first = { ...picked, end: playhead }
    const second = { ...picked, id: uid(), start: playhead }
    setClips(previous => previous.flatMap(clip => clip.id === picked.id ? [first, second] : [clip]))
    setSelected(second.id); setResult(null)
  }

  function shift(index, amount) {
    const next = [...clips], other = index + amount
    if (other < 0 || other >= next.length) return
    ;[next[index], next[other]] = [next[other], next[index]]
    setClips(next); setResult(null)
  }

  function moveClip(fromId, toId) {
    if (!fromId || fromId === toId) return
    setClips(previous => {
      const next = [...previous], from = next.findIndex(clip => clip.id === fromId), to = next.findIndex(clip => clip.id === toId)
      if (from < 0 || to < 0) return previous
      const [item] = next.splice(from, 1)
      next.splice(to, 0, item)
      return next
    })
    setResult(null)
  }

  function deleteClip(id) {
    setClips(previous => previous.filter(clip => clip.id !== id))
    if (selected === id) setSelected('')
    setResult(null)
  }

  async function exportFile() {
    if (busy) return
    onError('')
    if (!clips.length || clips.some(clip => clip.end == null || clip.end - clip.start < .1)) {
      onError('Hãy chờ tải thời lượng và giữ mỗi đoạn dài ít nhất 0,1 giây.'); return
    }
    const used = assets.filter(asset => clips.some(clip => clip.sourceId === asset.id))
    const indices = new Map(used.map((asset, index) => [asset.id, index]))
    const form = new FormData()
    used.forEach(asset => form.append('files', asset.file))
    form.append('timeline', JSON.stringify({ clips: clips.map(clip => ({ source: indices.get(clip.sourceId), start: clip.start, end: clip.end, volume: clip.volume, muted: clip.muted })), overlay_at: Number(overlayAt), overlay_volume: Number(overlayVolume) }))
    if (overlay) form.append('overlay', overlay.file)
    setBusy(true); setResult(null)
    try {
      const response = await api('/editor/export', { method: 'POST', body: form })
      setResult(response); onNotice(`Đã xuất ${response.format.toUpperCase()} dài ${stamp(response.duration)}.`)
    } catch (error) { onError(error.message) } finally { setBusy(false) }
  }

  const Media = activeAsset?.video ? 'video' : 'audio'
  return <div className="editor-page">
    <div className="page-heading compact"><div><div className="eyebrow"><span/> DỰNG VIDEO · CHỈNH AUDIO</div><h1>Chỉnh từng đoạn. <em>Kể trọn câu chuyện.</em></h1><p>Đưa MP3, MP4 hoặc file media vào timeline; cắt, tách, xếp lại và chèn giọng AI trước khi xuất file.</p></div></div>
    <div className="editor-top">
      <section className="panel editor-library"><div className="panel-title"><span className="panel-icon purple"><UploadCloud size={20}/></span><div><h3>Kho media</h3><p>Giữ nguyên toàn bộ thời lượng file nguồn</p></div></div>
        <label className="editor-add"><Plus size={19}/> Thêm audio hoặc video<input type="file" accept="audio/*,video/*,.mkv,.opus" multiple onChange={event => { addFiles(event.target.files); event.target.value = '' }}/></label>
        <div className="editor-assets">{assets.length ? assets.map(asset => <div className="editor-asset" key={asset.id}><span className="editor-asset-icon">{asset.video ? <Film size={17}/> : <Music2 size={17}/>}</span><span className="editor-asset-name"><strong title={asset.file.name}>{asset.file.name}</strong><small>{asset.duration ? stamp(asset.duration) : 'Đang đọc thời lượng'} · {(asset.file.size / 1048576).toFixed(1)} MB</small></span><button type="button" title="Xóa file khỏi dự án" onClick={() => removeAsset(asset.id)}><Trash2 size={17}/></button>{asset.video ? <video className="metadata-probe" src={asset.url} preload="metadata" onLoadedMetadata={event => metadata(asset.id, event.currentTarget.duration)}/> : <audio className="metadata-probe" src={asset.url} preload="metadata" onLoadedMetadata={event => metadata(asset.id, event.currentTarget.duration)}/>}</div>) : <p className="editor-empty-assets">Chưa có file. Có thể thêm nhiều file và xếp lại ở timeline.</p>}</div>
      </section>
      <section className="panel editor-preview"><div className="panel-title"><span className="panel-icon blue"><Play size={20}/></span><div><h3>Xem trước đoạn đang chọn</h3><p>{activeAsset?.file.name || 'Chọn một đoạn trong timeline'}</p></div></div>
        <div className={`editor-screen ${activeAsset?.video ? '' : 'audio-screen'}`}>{activeAsset ? <Media key={activeAsset.id} ref={mediaRef} src={activeAsset.url} controls preload="metadata" onLoadedMetadata={event => { event.currentTarget.currentTime = picked?.start || 0 }} onTimeUpdate={event => { setPlayhead(event.currentTarget.currentTime); if (picked?.end != null && event.currentTarget.currentTime >= picked.end) event.currentTarget.pause() }}/> : <div className="editor-placeholder"><AudioLines size={46}/><span>Khung xem trước</span></div>}</div>
        {picked && <div className="editor-playbar"><span>{stamp(playhead)}</span><input aria-label="Di chuyển vị trí xem trước" type="range" min={picked.start} max={picked.end || picked.start + 1} step="0.01" value={Math.min(Math.max(playhead, picked.start), picked.end || picked.start + 1)} onChange={event => { const time = Number(event.target.value); setPlayhead(time); if (mediaRef.current) mediaRef.current.currentTime = time }}/><span>{stamp(picked.end || 0)}</span></div>}
      </section>
    </div>
    <section className="panel editor-timeline"><div className="editor-timeline-head"><div><strong>Timeline</strong><small>{clips.length} đoạn · {stamp(total)} tổng thời lượng</small></div><button type="button" className="button outline" disabled={!picked || !picked.end} onClick={split}><Scissors size={16}/> Tách tại vị trí phát</button></div>
      <div className="editor-ruler"><span>00:00</span><span>{stamp(total / 4)}</span><span>{stamp(total / 2)}</span><span>{stamp(total * .75)}</span><span>{stamp(total)}</span></div>
      <div className="editor-track-label"><Film size={15}/> VIDEO / AUDIO GỐC · KÉO ĐỂ XẾP LẠI</div><div className="editor-track">{clips.length ? clips.map(clip => { const asset = assets.find(item => item.id === clip.sourceId); return <button type="button" key={clip.id} draggable onDragStart={event => event.dataTransfer.setData('text/plain', clip.id)} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); moveClip(event.dataTransfer.getData('text/plain'), clip.id) }} style={{ flexGrow: Math.max(1, (clip.end || clip.start + 1) - clip.start) }} className={`editor-block ${selected === clip.id ? 'selected' : ''} ${asset?.video ? 'video-block' : 'audio-block'}`} onClick={() => setSelected(clip.id)}><span>{asset?.video ? <Film size={14}/> : <Music2 size={14}/>} {asset?.file.name}</span><small>{stamp(Math.max(0, (clip.end || 0) - clip.start))}</small></button> }) : <p>Chưa có đoạn nào. Thêm file để bắt đầu.</p>}</div>
      <div className="editor-track-label"><Music2 size={15}/> ÂM THANH CHÈN THÊM</div><div className="editor-overlay-track">{overlay ? <><span>{overlay.file.name}</span><button type="button" onClick={() => { URL.revokeObjectURL(overlay.url); setOverlay(null); setResult(null) }}><Trash2 size={15}/> Xóa lớp audio</button></> : <span>Thêm nhạc nền hoặc giọng AI ở phần bên dưới.</span>}</div>
    </section>
    <div className="editor-bottom"><section className="panel editor-inspector"><div className="panel-title"><span className="panel-icon purple"><Scissors size={19}/></span><div><h3>Chỉnh đoạn đã chọn</h3><p>{activeAsset?.file.name || 'Chọn một đoạn trên timeline'}</p></div></div>{picked ? <><div className="editor-inspector-grid"><label>Bắt đầu (giây)<input type="number" min="0" max={picked.end || undefined} step="0.1" value={picked.start} onChange={event => updateClip(picked.id, { start: Number(event.target.value) })}/></label><label>Kết thúc (giây)<input type="number" min={picked.start} max={activeAsset?.duration || undefined} step="0.1" value={picked.end ?? ''} onChange={event => updateClip(picked.id, { end: Number(event.target.value) })}/></label><label>Âm lượng <b>{Math.round(picked.volume * 100)}%</b><input type="range" min="0" max="2" step="0.05" value={picked.volume} onChange={event => updateClip(picked.id, { volume: Number(event.target.value) })}/></label><label className="editor-mute"><input type="checkbox" checked={picked.muted} onChange={event => updateClip(picked.id, { muted: event.target.checked })}/> Tắt tiếng đoạn này</label></div><div className="editor-clip-actions"><button type="button" onClick={() => shift(clips.findIndex(item => item.id === picked.id), -1)}><ChevronLeft size={15}/> Đưa lên trước</button><button type="button" onClick={() => shift(clips.findIndex(item => item.id === picked.id), 1)}>Đưa ra sau <ChevronRight size={15}/></button><button type="button" className="danger" onClick={() => deleteClip(picked.id)}><Trash2 size={15}/> Xóa đoạn</button></div></> : <p className="fine-print">Chọn đoạn để cắt thời điểm, đổi thứ tự hoặc tắt tiếng.</p>}</section>
      <section className="panel editor-music"><div className="panel-title"><span className="panel-icon blue"><Music2 size={19}/></span><div><h3>Nhạc nền / giọng AI</h3><p>Một lớp audio đặt tại thời điểm bạn chọn</p></div></div><label className="editor-add secondary"><Plus size={17}/> {overlay ? 'Đổi lớp audio' : 'Chọn file audio để chèn'}<input type="file" accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg,.opus" onChange={event => { const file = event.target.files?.[0]; if (file) { setOverlay(previous => { if (previous) URL.revokeObjectURL(previous.url); return { file, url: URL.createObjectURL(file) } }); setResult(null) } event.target.value = '' }}/></label>{overlay && <><audio controls src={overlay.url} className="editor-overlay-audio"/><div className="editor-inspector-grid"><label>Chèn tại giây thứ<input type="number" min="0" max={total || undefined} step="0.1" value={overlayAt} onChange={event => setOverlayAt(event.target.value)}/></label><label>Âm lượng <b>{Math.round(overlayVolume * 100)}%</b><input type="range" min="0" max="2" step="0.05" value={overlayVolume} onChange={event => setOverlayVolume(Number(event.target.value))}/></label></div></>}</section></div>
    <section className="panel editor-export"><div><strong>Xuất bản dựng</strong><p>Có hình sẽ xuất MP4; chỉ có audio sẽ xuất WAV. File nguồn được xử lý đầy đủ, không giới hạn ở 8 giây mẫu giọng.</p></div><button type="button" className="button primary" disabled={busy || !clips.length} onClick={exportFile}>{busy ? 'Đang dựng và xuất file…' : <><Download size={17}/> Xuất MP4 / WAV</>}</button>{result && <div className="editor-output">{result.format === 'mp4' ? <video src={result.url} controls/> : <audio src={result.url} controls/>}<a href={result.url} download><Download size={17}/> Tải {result.format.toUpperCase()}</a></div>}</section>
  </div>
}

import React, { useEffect, useState } from 'react'

export default function TrainingReview({ api, voiceId, onApproved, onError }) {
  const [offset, setOffset] = useState(0)
  const [review, setReview] = useState(null)
  const [drafts, setDrafts] = useState({})
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let live = true
    api(`/voices/${voiceId}/training-review?offset=${offset}&limit=20`)
      .then(data => { if (live) { setReview(data); setDrafts(Object.fromEntries(data.clips.map(clip => [clip.file_name, clip]))) } })
      .catch(error => { if (live) onError(error.message) })
    return () => { live = false }
  }, [voiceId, offset])

  async function save(clip) {
    try {
      await api(`/voices/${voiceId}/training-review/${clip.file_name}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: clip.text, include: clip.include }),
      })
    } catch (error) { onError(error.message); throw error }
  }

  async function approve() {
    setBusy(true)
    try {
      await Promise.all(review.clips.filter(row => {
        const draft = drafts[row.file_name]
        return draft && (draft.text !== row.text || draft.include !== row.include)
      }).map(row => save(drafts[row.file_name])))
      const result = await api(`/voices/${voiceId}/training-review/approve`, { method: 'POST' })
      onApproved(result.training)
    } catch (error) { onError(error.message) } finally { setBusy(false) }
  }

  if (!review) return <p>Đang tải các đoạn lời nói để duyệt…</p>
  return <div className="training-review">
    <p><strong>Duyệt dữ liệu trước khi train · {review.total} đoạn</strong></p>
    <p className="fine-print">Nghe từng đoạn, sửa lời nhận dạng sai và bỏ tiếng người khác hoặc đoạn nhiễu. Nút duyệt xác nhận các đoạn chưa sửa trên mọi trang vẫn được giữ.</p>
    {review.clips.map(row => {
      const clip = drafts[row.file_name] || row
      return <div className="training-review-row" key={row.file_name}>
        <audio src={`/api/voices/${voiceId}/training-review/${row.file_name}`} controls preload="none" aria-label={`Nghe đoạn ${row.file_name}`}/>
        <label><input type="checkbox" checked={clip.include} onChange={event => {
          const next = { ...clip, include: event.target.checked }
          setDrafts(previous => ({ ...previous, [row.file_name]: next })); save(next).catch(() => {})
        }}/> Giữ</label>
        <input className="text-input" aria-label={`Lời đoạn ${row.file_name}`} value={clip.text}
          onChange={event => setDrafts(previous => ({ ...previous, [row.file_name]: { ...clip, text: event.target.value } }))}
          onBlur={() => { if (clip.text !== row.text) save(clip).catch(() => {}) }}/>
        <small>{row.source_start.toFixed(1)}–{row.source_end.toFixed(1)}s</small>
      </div>
    })}
    <div className="training-review-actions">
      <button type="button" className="button outline" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>← Trước</button>
      <span>{offset + 1}–{Math.min(offset + 20, review.total)} / {review.total}</span>
      <button type="button" className="button outline" disabled={offset + 20 >= review.total} onClick={() => setOffset(offset + 20)}>Tiếp →</button>
    </div>
    <button type="button" className="button primary full" disabled={busy} onClick={approve}>{busy ? 'Đang xếp hàng…' : 'Đã duyệt clip · bắt đầu train'}</button>
  </div>
}

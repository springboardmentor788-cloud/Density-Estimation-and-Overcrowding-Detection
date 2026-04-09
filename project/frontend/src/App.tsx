import { Camera, ImagePlus, LoaderCircle, Play, Radar, Upload } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { MediaCompare } from './components/MediaCompare'
import { MetricCard } from './components/MetricCard'
import { dashboardCopy, thresholdPresets } from './data/mockData'
import { resolveAssetUrl } from './api/client'
import { useInference } from './hooks/useInference'
import { useModels } from './hooks/useModels'
import { useWebcam } from './hooks/useWebcam'

type TabKey = 'live' | 'image' | 'video'

export default function App() {
  const { models, selectedModel, loading: modelsLoading, error: modelsError, setSelectedPath } = useModels()
  const { imageResult, videoResult, imageLoading, videoLoading, error, runImageInference, runVideoInference } = useInference()
  const { videoRef, isStreaming, startWebcam, stopWebcam, captureFrame } = useWebcam()

  const [tab, setTab] = useState<TabKey>('live')
  const [threshold, setThreshold] = useState(130)
  const [resizeWidth, setResizeWidth] = useState(1280)
  const [sampleFps, setSampleFps] = useState(8)
  const [calibrationPath, setCalibrationPath] = useState('/home/ishouriya/Desktop/springboard/try/outputs/count_calibration_partA_push_v1_mae.json')
  const [preferCuda, setPreferCuda] = useState(true)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [videoPreview, setVideoPreview] = useState<string | null>(null)
  const [liveOverlay, setLiveOverlay] = useState<string | null>(null)
  const [liveDensity, setLiveDensity] = useState<string | null>(null)
  const [liveCount, setLiveCount] = useState<number | null>(null)
  const [liveStatus, setLiveStatus] = useState<string>('SAFE')
  const [isLiveRunning, setIsLiveRunning] = useState(false)

  const currentModelPath = selectedModel?.path ?? models[0]?.path ?? ''
  const currentModelLabel = selectedModel?.label ?? 'Loading model catalog...'

  useEffect(() => {
    if (!selectedModel && models[0]) {
      setSelectedPath(models[0].path)
    }
  }, [models, selectedModel, setSelectedPath])

  useEffect(() => {
    if (!isLiveRunning) return

    let cancelled = false
    const loop = async () => {
      try {
        const frameBlob = await captureFrame()
        if (cancelled) return
        const result = await runImageInference({
          file: new File([frameBlob], `live-${Date.now()}.jpg`, { type: 'image/jpeg' }),
          checkpointPath: currentModelPath,
          calibrationPath,
          threshold,
          resizeWidth,
        })
        if (cancelled) return
        setLiveOverlay(result.overlay_image)
        setLiveDensity(result.density_image)
        setLiveCount(result.count)
        setLiveStatus(result.status)
      } catch {
        // surfaced via hook error state; keep loop responsive
      }
    }

    const timer = window.setInterval(() => void loop(), 1800)
    void loop()
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [isLiveRunning, captureFrame, runImageInference, currentModelPath, calibrationPath, threshold, resizeWidth])

  const selectedPreset = useMemo(() => thresholdPresets.find((value) => value === threshold) ?? null, [threshold])

  const onImageUpload = async (file: File) => {
    const result = await runImageInference({
      file,
      checkpointPath: currentModelPath,
      calibrationPath,
      threshold,
      resizeWidth,
    })
    setImagePreview(file.name)
    return result
  }

  const onVideoUpload = async (file: File) => {
    const result = await runVideoInference({
      file,
      checkpointPath: currentModelPath,
      calibrationPath,
      threshold,
      resizeWidth,
      sampleFps,
    })
    setVideoPreview(file.name)
    return result
  }

  const videoOverlayUrl = videoResult ? resolveAssetUrl(videoResult.overlay_url) : null
  const videoDensityUrl = videoResult ? resolveAssetUrl(videoResult.density_url) : null

  const headlineMetrics = [
    { label: 'Selected model MAE', value: selectedModel ? selectedModel.mae.toFixed(1) : '—', hint: 'validation score', tone: 'cyan' as const },
    { label: 'Crowd threshold', value: threshold.toString(), hint: selectedPreset ? 'preset selected' : 'custom threshold', tone: 'amber' as const },
    { label: 'Inference width', value: `${resizeWidth}px`, hint: 'resized for model', tone: 'emerald' as const },
    { label: 'Live sampling', value: `${sampleFps} fps`, hint: 'video analysis rate', tone: 'violet' as const },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar glass-panel">
        <div className="brand-block">
          <div className="brand-mark">DV</div>
          <div>
            <div className="brand-title">{dashboardCopy.brand}</div>
            <div className="brand-subtitle">Crowd Intelligence Studio</div>
          </div>
        </div>

        <div className="stack-gap">
          <div className="section-title">Model Catalog</div>
          <label className="control-label" htmlFor="modelSelect">Trained checkpoint</label>
          <select
            id="modelSelect"
            className="select-input"
            value={currentModelPath}
            onChange={(event) => setSelectedPath(event.target.value)}
          >
            {modelsLoading ? <option>Loading models...</option> : null}
            {models.map((model) => (
              <option key={model.path} value={model.path}>
                {model.label}
              </option>
            ))}
          </select>
          {modelsError ? <div className="error-pill">{modelsError}</div> : null}

          <div className="mini-list">
            {models.slice(0, 4).map((model) => (
              <button key={model.path} className={`mini-pill ${model.path === currentModelPath ? 'active' : ''}`} onClick={() => setSelectedPath(model.path)}>
                <span>{model.mae.toFixed(1)} MAE</span>
                <strong>epoch {model.epoch}</strong>
              </button>
            ))}
          </div>
        </div>

        <div className="stack-gap">
          <div className="section-title">Inference Controls</div>
          <label className="control-label">Calibration file</label>
          <input className="text-input" value={calibrationPath} onChange={(event) => setCalibrationPath(event.target.value)} />

          <label className="control-label">Threshold presets</label>
          <div className="preset-grid">
            {thresholdPresets.map((value) => (
              <button key={value} className={`preset-chip ${threshold === value ? 'active' : ''}`} onClick={() => setThreshold(value)}>
                {value}
              </button>
            ))}
          </div>

          <label className="control-label">Overcrowding threshold</label>
          <input className="slider-input" type="range" min={20} max={1000} step={5} value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />

          <label className="control-label">Resize width</label>
          <input className="slider-input" type="range" min={320} max={1920} step={32} value={resizeWidth} onChange={(event) => setResizeWidth(Number(event.target.value))} />

          <label className="control-label">Video sampling FPS</label>
          <input className="slider-input" type="range" min={1} max={30} step={1} value={sampleFps} onChange={(event) => setSampleFps(Number(event.target.value))} />

          <label className="switch-row">
            <input type="checkbox" checked={preferCuda} onChange={(event) => setPreferCuda(event.target.checked)} />
            <span>Use CUDA if available</span>
          </label>
        </div>

        <div className="sidebar-footer">
          <div className="section-title">Quick Readout</div>
          <div className="footer-chip">Selected model: {selectedModel ? selectedModel.label : '—'}</div>
          <div className="footer-chip">Calibration: {calibrationPath ? 'enabled' : 'disabled'}</div>
          <div className="footer-chip">Backend: {preferCuda ? 'GPU preferred' : 'CPU only'}</div>
        </div>
      </aside>

      <main className="content-area">
        <header className="hero glass-panel">
          <div>
            <div className="eyebrow">{dashboardCopy.brand} / React control room</div>
            <h1>{dashboardCopy.title}</h1>
            <p>{dashboardCopy.subtitle}</p>
          </div>
          <div className="hero-actions">
            <button className={`tab-button ${tab === 'live' ? 'active' : ''}`} onClick={() => setTab('live')}>
              <Camera size={16} /> Live
            </button>
            <button className={`tab-button ${tab === 'image' ? 'active' : ''}`} onClick={() => setTab('image')}>
              <ImagePlus size={16} /> Image
            </button>
            <button className={`tab-button ${tab === 'video' ? 'active' : ''}`} onClick={() => setTab('video')}>
              <Play size={16} /> Video
            </button>
          </div>
        </header>

        <section className="metrics-grid">
          {headlineMetrics.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </section>

        {error ? <div className="error-banner">{error}</div> : null}

        {tab === 'live' ? (
          <section className="glass-panel panel-stack">
            <div className="panel-header">
              <div>
                <div className="section-title">Live camera capture</div>
                <p className="section-copy">{dashboardCopy.liveHint}</p>
              </div>
              <div className="panel-actions">
                <button className="primary-button" onClick={async () => { await startWebcam(); setIsLiveRunning(true) }}>
                  <Play size={16} /> Start analysis
                </button>
                <button className="secondary-button" onClick={() => { setIsLiveRunning(false); stopWebcam() }}>
                  Stop
                </button>
              </div>
            </div>

            <div className="live-layout">
              <div className="media-panel camera-panel">
                <div className="media-heading">Camera preview</div>
                <video ref={videoRef} className="camera-preview" autoPlay playsInline muted />
              </div>
              <div className="media-panel">
                <div className="media-heading">Overlay output</div>
                {liveOverlay ? <img className="media-fit" src={liveOverlay} alt="live overlay output" /> : <div className="media-placeholder">Start analysis to render the overlay</div>}
              </div>
              <div className="media-panel">
                <div className="media-heading">Density output</div>
                {liveDensity ? <img className="media-fit" src={liveDensity} alt="live density output" /> : <div className="media-placeholder">Heatmap output appears here</div>}
              </div>
            </div>

            <div className="live-meta">
              <MetricCard label="Current count" value={liveCount ? liveCount.toFixed(1) : '—'} hint={liveStatus} tone={liveStatus === 'OVERCROWDED' ? 'amber' : 'emerald'} />
              <MetricCard label="Analysis mode" value={isLiveRunning ? 'Running' : 'Idle'} hint="browser webcam sampling" tone="violet" />
              <MetricCard label="Model path" value={selectedModel ? selectedModel.path.split('/').pop() ?? 'model' : '—'} hint="active checkpoint" tone="cyan" />
            </div>
          </section>
        ) : null}

        {tab === 'image' ? (
          <section className="glass-panel panel-stack">
            <div className="panel-header">
              <div>
                <div className="section-title">Image analysis</div>
                <p className="section-copy">Upload a still image and receive embedded crowd overlay plus density-only output.</p>
              </div>
              <label className="primary-button file-button">
                <Upload size={16} /> Upload image
                <input
                  type="file"
                  accept="image/*"
                  onChange={async (event) => {
                    const file = event.target.files?.[0]
                    if (!file) return
                    const result = await onImageUpload(file)
                    setImagePreview(file.name)
                    setLiveOverlay(result.overlay_image)
                    setLiveDensity(result.density_image)
                  }}
                />
              </label>
            </div>

            <div className="image-banner">{imagePreview ? `Loaded: ${imagePreview}` : 'No image selected yet.'}</div>
            <MediaCompare
              kind="image"
              leftTitle="Overlay image"
              rightTitle="Density-only image"
              leftSrc={imageResult ? imageResult.overlay_image : null}
              rightSrc={imageResult ? imageResult.density_image : null}
            />
          </section>
        ) : null}

        {tab === 'video' ? (
          <section className="glass-panel panel-stack">
            <div className="panel-header">
              <div>
                <div className="section-title">Video analysis</div>
                <p className="section-copy">{dashboardCopy.videoHint}</p>
              </div>
              <label className="primary-button file-button">
                <Upload size={16} /> Upload video
                <input
                  type="file"
                  accept="video/*"
                  onChange={async (event) => {
                    const file = event.target.files?.[0]
                    if (!file) return
                    await onVideoUpload(file)
                    setVideoPreview(file.name)
                  }}
                />
              </label>
            </div>

            {videoPreview ? <div className="image-banner">Loaded: {videoPreview}</div> : null}

            <div className="video-stats-grid">
              <MetricCard label="Frames processed" value={videoResult ? videoResult.frames.toFixed(0) : '—'} hint="processed frames" tone="cyan" />
              <MetricCard label="Average count" value={videoResult ? videoResult.count_mean.toFixed(1) : '—'} hint="mean per sampled frame" tone="emerald" />
              <MetricCard label="Peak count" value={videoResult ? videoResult.count_max.toFixed(1) : '—'} hint="highest sampled frame" tone="amber" />
              <MetricCard label="Floor count" value={videoResult ? videoResult.count_min.toFixed(1) : '—'} hint="lowest sampled frame" tone="violet" />
            </div>

            <div className="compare-grid compare-video-grid">
              <section className="media-panel">
                <div className="media-heading">Overlay video</div>
                {videoOverlayUrl ? <video className="media-fit" src={videoOverlayUrl} controls playsInline /> : <div className="media-placeholder">Overlay output video will appear here</div>}
              </section>
              <section className="media-panel">
                <div className="media-heading">Density map video</div>
                {videoDensityUrl ? <video className="media-fit" src={videoDensityUrl} controls playsInline /> : <div className="media-placeholder">Density output video will appear here</div>}
              </section>
            </div>

            <div className="action-row">
              <button className="secondary-button" disabled={!videoOverlayUrl} onClick={() => videoOverlayUrl && window.open(videoOverlayUrl, '_blank')}>Open overlay file</button>
              <button className="secondary-button" disabled={!videoDensityUrl} onClick={() => videoDensityUrl && window.open(videoDensityUrl, '_blank')}>Open density file</button>
            </div>

            {videoLoading ? <div className="loading-note"><LoaderCircle size={16} className="spin" /> Processing video...</div> : null}
          </section>
        ) : null}
      </main>
    </div>
  )
}

export interface ModelOption {
  label: string
  path: string
  mae: number
  rmse: number
  epoch: number
}

export interface ImageInferenceResponse {
  count: number
  status: string
  overlay_image: string
  density_image: string
}

export interface VideoInferenceResponse {
  session_id: string
  frames: number
  count_mean: number
  count_max: number
  count_min: number
  overlay_url: string
  density_url: string
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

export async function fetchModels(): Promise<ModelOption[]> {
  const response = await fetch(apiUrl('/api/models'))
  if (!response.ok) {
    throw new Error('Failed to load models')
  }
  return (await response.json()) as ModelOption[]
}

export async function inferImage(params: {
  file: File
  checkpointPath: string
  calibrationPath?: string
  threshold: number
  resizeWidth: number
  preferCuda?: boolean
}): Promise<ImageInferenceResponse> {
  const formData = new FormData()
  formData.append('file', params.file)
  formData.append('checkpoint_path', params.checkpointPath)
  formData.append('threshold', String(params.threshold))
  formData.append('resize_width', String(params.resizeWidth))
  formData.append('calibration_path', params.calibrationPath ?? '')
  formData.append('prefer_cuda', String(params.preferCuda ?? true))

  const response = await fetch(apiUrl('/api/infer/image'), {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error('Image inference failed')
  }
  return (await response.json()) as ImageInferenceResponse
}

export async function inferVideo(params: {
  file: File
  checkpointPath: string
  calibrationPath?: string
  threshold: number
  resizeWidth: number
  sampleFps: number
  preferCuda?: boolean
}): Promise<VideoInferenceResponse> {
  const formData = new FormData()
  formData.append('file', params.file)
  formData.append('checkpoint_path', params.checkpointPath)
  formData.append('threshold', String(params.threshold))
  formData.append('resize_width', String(params.resizeWidth))
  formData.append('sample_fps', String(params.sampleFps))
  formData.append('calibration_path', params.calibrationPath ?? '')
  formData.append('prefer_cuda', String(params.preferCuda ?? true))

  const response = await fetch(apiUrl('/api/infer/video'), {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error('Video inference failed')
  }
  return (await response.json()) as VideoInferenceResponse
}

export function resolveAssetUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('data:')) {
    return path
  }
  return `${API_BASE}${path}`
}

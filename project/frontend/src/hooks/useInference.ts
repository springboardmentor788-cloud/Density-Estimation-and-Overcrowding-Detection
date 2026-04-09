import { useState } from 'react'
import { inferImage, inferVideo, type ImageInferenceResponse, type VideoInferenceResponse } from '../api/client'

export interface UseInferenceResult {
  readonly imageResult: ImageInferenceResponse | null
  readonly videoResult: VideoInferenceResponse | null
  readonly imageLoading: boolean
  readonly videoLoading: boolean
  readonly error: string | null
  readonly runImageInference: (args: {
    file: File
    checkpointPath: string
    calibrationPath?: string
    threshold: number
    resizeWidth: number
  }) => Promise<ImageInferenceResponse>
  readonly runVideoInference: (args: {
    file: File
    checkpointPath: string
    calibrationPath?: string
    threshold: number
    resizeWidth: number
    sampleFps: number
  }) => Promise<VideoInferenceResponse>
  readonly clearError: () => void
}

export function useInference(): UseInferenceResult {
  const [imageResult, setImageResult] = useState<ImageInferenceResponse | null>(null)
  const [videoResult, setVideoResult] = useState<VideoInferenceResponse | null>(null)
  const [imageLoading, setImageLoading] = useState(false)
  const [videoLoading, setVideoLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runImageInference: UseInferenceResult['runImageInference'] = async (args) => {
    setError(null)
    setImageLoading(true)
    try {
      const result = await inferImage(args)
      setImageResult(result)
      return result
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : 'Image inference failed'
      setError(message)
      throw caught
    } finally {
      setImageLoading(false)
    }
  }

  const runVideoInference: UseInferenceResult['runVideoInference'] = async (args) => {
    setError(null)
    setVideoLoading(true)
    try {
      const result = await inferVideo(args)
      setVideoResult(result)
      return result
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : 'Video inference failed'
      setError(message)
      throw caught
    } finally {
      setVideoLoading(false)
    }
  }

  return {
    imageResult,
    videoResult,
    imageLoading,
    videoLoading,
    error,
    runImageInference,
    runVideoInference,
    clearError: () => setError(null),
  }
}

import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

export interface UseWebcamResult {
  readonly videoRef: RefObject<HTMLVideoElement>
  readonly isStreaming: boolean
  readonly startWebcam: () => Promise<void>
  readonly stopWebcam: () => void
  readonly captureFrame: () => Promise<Blob>
}

export function useWebcam(): UseWebcamResult {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  const stopWebcam = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
  }, [])

  const startWebcam = useCallback(async () => {
    if (streamRef.current) return
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    streamRef.current = stream
    if (videoRef.current) {
      videoRef.current.srcObject = stream
      await videoRef.current.play()
    }
    setIsStreaming(true)
  }, [])

  const captureFrame = useCallback(async () => {
    const video = videoRef.current
    if (!video) {
      throw new Error('Webcam is not ready')
    }
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 1280
    canvas.height = video.videoHeight || 720
    const context = canvas.getContext('2d')
    if (!context) {
      throw new Error('Could not create canvas context')
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    return await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error('Failed to capture webcam frame'))
          return
        }
        resolve(blob)
      }, 'image/jpeg', 0.92)
    })
  }, [])

  useEffect(() => () => stopWebcam(), [stopWebcam])

  return { videoRef, isStreaming, startWebcam, stopWebcam, captureFrame }
}

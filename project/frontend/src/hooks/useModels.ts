import { useEffect, useMemo, useState } from 'react'
import { fetchModels, type ModelOption } from '../api/client'

export interface UseModelsResult {
  readonly models: ModelOption[]
  readonly selectedModel: ModelOption | null
  readonly loading: boolean
  readonly error: string | null
  readonly setSelectedPath: (path: string) => void
}

export function useModels(): UseModelsResult {
  const [models, setModels] = useState<ModelOption[]>([])
  const [selectedPath, setSelectedPath] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        setLoading(true)
        const nextModels = await fetchModels()
        if (cancelled) return
        setModels(nextModels)
        setSelectedPath((current) => current || nextModels[0]?.path || '')
      } catch (caught) {
        if (cancelled) return
        setError(caught instanceof Error ? caught.message : 'Unable to load models')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [])

  const selectedModel = useMemo(
    () => models.find((model) => model.path === selectedPath) ?? null,
    [models, selectedPath],
  )

  return {
    models,
    selectedModel,
    loading,
    error,
    setSelectedPath,
  }
}

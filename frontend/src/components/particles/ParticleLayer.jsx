import { Suspense, lazy, useEffect, useState } from 'react'
import { useReducedMotion } from 'motion/react'

// three.js is large: load it only on the page that shows the particles.
const ParticleStory = lazy(() => import('./ParticleStory'))

function hasWebGL() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    return Boolean(window.WebGLRenderingContext && (canvas.getContext('webgl2') || canvas.getContext('webgl')))
  } catch {
    return false
  }
}

/** The particle animation, or `fallback` where WebGL is missing (old machines, tests). */
export function ParticleLayer({ className = '', fallback = null, onStage }) {
  const reducedMotion = useReducedMotion()
  const [webgl, setWebgl] = useState(null)
  useEffect(() => setWebgl(hasWebGL()), [])
  if (webgl === false) return fallback
  if (webgl === null) return null
  return (
    <div className={className} aria-hidden="true">
      <Suspense fallback={fallback}>
        <ParticleStory reducedMotion={Boolean(reducedMotion)} onStage={onStage} />
      </Suspense>
    </div>
  )
}

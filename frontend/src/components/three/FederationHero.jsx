import { Suspense, lazy, useEffect, useState } from 'react'
import { useReducedMotion } from 'motion/react'

// three.js is large: load it only when a page shows the scene, so other pages stay light.
const FederationScene = lazy(() => import('./FederationScene'))

function hasWebGL() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    return Boolean(window.WebGLRenderingContext && (canvas.getContext('webgl2') || canvas.getContext('webgl')))
  } catch {
    return false
  }
}

/**
 * The animated federation (hospitals around the coordinator). Falls back to `fallback`
 * (normally the static illustration) where WebGL is missing, e.g. old machines and tests.
 */
export function FederationHero({ hospitals, active = false, fallback = null, className = '', cameraZ }) {
  const reducedMotion = useReducedMotion()
  const [webgl, setWebgl] = useState(null)
  useEffect(() => setWebgl(hasWebGL()), [])

  if (webgl === false) return fallback
  return (
    <div className={className}>
      {webgl ? (
        <Suspense fallback={fallback}>
          <FederationScene hospitals={hospitals} active={active} reducedMotion={Boolean(reducedMotion)} cameraZ={cameraZ} />
        </Suspense>
      ) : null}
    </div>
  )
}

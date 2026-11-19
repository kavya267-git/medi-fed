import { Suspense, lazy, useEffect, useState } from 'react'
import { useReducedMotion } from 'motion/react'

// three.js is large: load it only where a page shows the shapes, so other pages stay light.
const FloatingShapes = lazy(() => import('./FloatingShapes'))

function hasWebGL() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    return Boolean(window.WebGLRenderingContext && (canvas.getContext('webgl2') || canvas.getContext('webgl')))
  } catch {
    return false
  }
}

/** Floating 3D shapes behind a hero area. Renders nothing where WebGL is missing (old machines, tests). */
export function FloatingLayer({ variant = 'hero', className = '' }) {
  const reducedMotion = useReducedMotion()
  const [webgl, setWebgl] = useState(false)
  useEffect(() => setWebgl(hasWebGL()), [])
  if (!webgl) return null
  return (
    <div className={className} aria-hidden="true">
      <Suspense fallback={null}>
        <FloatingShapes variant={variant} reducedMotion={Boolean(reducedMotion)} />
      </Suspense>
    </div>
  )
}

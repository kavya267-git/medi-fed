import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

/*
  A cloud of glowing particles that reshapes itself, telling the project's story in four forms:
    1. four hospital clusters, with streams flowing to the middle (local training, updates sent),
    2. a sphere (the averaged global model),
    3. a tree growing upward (the shared model learning round after round),
    4. a cube (the model packaged and sent back to every hospital).
  Every particle has a position in each form; the vertex shader blends between neighbouring forms on the GPU.
  Ripple rings pulse on the ground below. Decoration only: hidden from assistive technology.
*/

const SHAPES = 4
const HOLD = 3.6 // seconds a form is held
const MOVE = 2.0 // seconds a change takes

function rand(seed) {
  // Deterministic pseudo-random numbers, so the picture is the same on every visit.
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

function buildForms(count) {
  const r = rand(7)
  const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r())
  const forms = Array.from({ length: SHAPES }, () => new Float32Array(count * 3))
  const seeds = new Float32Array(count)
  const corners = [
    [-2.4, 0.2, -1.2],
    [2.4, 0.2, -1.2],
    [-2.4, 0.2, 1.4],
    [2.4, 0.2, 1.4],
  ]
  for (let i = 0; i < count; i += 1) {
    seeds[i] = r()
    const k = i * 3

    // 1. Hospitals: four dense clusters, a share of particles strung along arcs to the centre.
    const c = corners[i % 4]
    if (r() < 0.62) {
      forms[0].set([c[0] + gauss() * 0.28, c[1] + Math.abs(gauss()) * 0.45, c[2] + gauss() * 0.28], k)
    } else {
      const t = r()
      const lift = Math.sin(Math.PI * t) * 1.1
      forms[0].set([c[0] * (1 - t) + gauss() * 0.04, c[1] * (1 - t) + 0.9 * t + lift + gauss() * 0.04, c[2] * (1 - t) + gauss() * 0.04], k)
    }

    // 2. Global model: a sphere, dense at the surface with a softer core.
    const u = r() * 2 - 1
    const phi = r() * Math.PI * 2
    const rad = r() < 0.8 ? 1.55 + gauss() * 0.04 : Math.cbrt(r()) * 1.4
    const s = Math.sqrt(1 - u * u)
    forms[1].set([Math.cos(phi) * s * rad, 1.2 + u * rad, Math.sin(phi) * s * rad], k)

    // 3. Tree: a trunk rising from the ground that opens into a wide canopy.
    const h = Math.pow(r(), 0.8)
    const y = -0.9 + h * 3.6
    const spread = h < 0.35 ? 0.12 + h * 0.3 : 0.2 + Math.pow((h - 0.35) / 0.65, 1.6) * 2.4
    const a = r() * Math.PI * 2
    const rr = spread * Math.sqrt(r())
    forms[2].set([Math.cos(a) * rr + gauss() * 0.05, y + gauss() * 0.06, Math.sin(a) * rr + gauss() * 0.05], k)

    // 4. Cube: particles on the faces, denser along the edges.
    const size = 1.45
    const pick = () => (r() * 2 - 1) * size
    const face = Math.floor(r() * 6)
    const edge = r() < 0.45
    let p = [pick(), pick(), pick()]
    const axis = face % 3
    p[axis] = face < 3 ? size : -size
    if (edge) p[(axis + 1) % 3] = r() < 0.5 ? size : -size
    forms[3].set([p[0], p[1] + 1.2, p[2]], k)
  }
  return { forms, seeds }
}

const vertexShader = /* glsl */ `
  attribute vec3 formA;
  attribute vec3 formB;
  attribute vec3 formC;
  attribute vec3 formD;
  attribute float seed;
  uniform float uTime;
  uniform float uMorph;
  uniform float uPixelRatio;
  varying float vSeed;
  varying float vGlow;

  vec3 formAt(float i) {
    if (i < 0.5) return formA;
    if (i < 1.5) return formB;
    if (i < 2.5) return formC;
    return formD;
  }

  void main() {
    float from = floor(uMorph);
    float t = uMorph - from;
    // Each particle leaves a little earlier or later than its neighbours, so changes ripple through the cloud.
    float local = smoothstep(0.0, 1.0, clamp((t - seed * 0.35) / 0.65, 0.0, 1.0));
    vec3 p = mix(formAt(mod(from, 4.0)), formAt(mod(from + 1.0, 4.0)), local);
    // Gentle drift, so the cloud breathes even while a form is held.
    p += 0.035 * vec3(sin(uTime * 0.9 + seed * 40.0), cos(uTime * 0.7 + seed * 23.0), sin(uTime * 0.8 + seed * 61.0));
    // Mid-transition the particles lift and scatter slightly, like the reference effect.
    p += vec3(0.0, sin(local * 3.14159) * 0.25, 0.0) * (0.5 + seed);

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float size = mix(1.4, 3.6, pow(seed, 3.0));
    gl_PointSize = size * uPixelRatio * (7.0 / -mv.z);
    vSeed = seed;
    vGlow = 0.55 + 0.45 * sin(uTime * 2.0 + seed * 30.0);
  }
`

const fragmentShader = /* glsl */ `
  varying float vSeed;
  varying float vGlow;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float alpha = smoothstep(0.5, 0.0, d);
    vec3 deep = vec3(0.16, 0.36, 0.95);
    vec3 cyan = vec3(0.45, 0.78, 1.0);
    vec3 white = vec3(0.92, 0.96, 1.0);
    vec3 color = mix(deep, cyan, vSeed);
    color = mix(color, white, step(0.965, vSeed));
    gl_FragColor = vec4(color * (0.75 + 0.5 * vGlow), alpha * 0.85);
  }
`

function Particles({ count, reducedMotion, onStage }) {
  const stage = useRef(-1)
  const material = useRef()
  const group = useRef()
  const { forms, seeds } = useMemo(() => buildForms(count), [count])
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(forms[0], 3))
    g.setAttribute('formA', new THREE.BufferAttribute(forms[0], 3))
    g.setAttribute('formB', new THREE.BufferAttribute(forms[1], 3))
    g.setAttribute('formC', new THREE.BufferAttribute(forms[2], 3))
    g.setAttribute('formD', new THREE.BufferAttribute(forms[3], 3))
    g.setAttribute('seed', new THREE.BufferAttribute(seeds, 1))
    return g
  }, [forms, seeds])
  const uniforms = useMemo(
    () => ({ uTime: { value: 0 }, uMorph: { value: 0 }, uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) } }),
    [],
  )

  useFrame((state, delta) => {
    // Write to the material's own uniforms: React Three Fiber copies the initial object into the material.
    const u = material.current.uniforms
    const time = state.clock.elapsedTime
    u.uTime.value = time
    if (!reducedMotion) {
      // Hold each form, then move to the next: morph runs 0 -> 4 and wraps.
      const cycle = HOLD + MOVE
      const step = Math.floor(time / cycle)
      const within = time - step * cycle
      u.uMorph.value = (step % SHAPES) + Math.max(0, (within - HOLD) / MOVE)
      // The form now on screen (it counts as the next one once the change is half done).
      const shown = (step + (within - HOLD > MOVE / 2 ? 1 : 0)) % SHAPES
      if (shown !== stage.current) {
        stage.current = shown
        onStage?.(shown)
      }
      group.current.rotation.y += delta * 0.12
    }
    const tiltX = state.pointer.y * 0.15
    group.current.rotation.x += (tiltX - group.current.rotation.x) * 0.04
  })

  return (
    <group ref={group}>
      <points geometry={geometry} frustumCulled={false}>
        <shaderMaterial
          ref={material}
          uniforms={uniforms}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
    </group>
  )
}

const rippleVertex = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`
const rippleFragment = /* glsl */ `
  uniform float uTime;
  varying vec2 vUv;
  void main() {
    float d = length(vUv - 0.5) * 2.0;
    float rings = 0.5 + 0.5 * sin(d * 38.0 - uTime * 2.4);
    rings = pow(rings, 14.0);
    float fade = smoothstep(1.0, 0.15, d) * smoothstep(0.0, 0.12, d);
    vec3 color = vec3(0.22, 0.48, 1.0);
    gl_FragColor = vec4(color, rings * fade * 0.55);
  }
`

function Ripples() {
  const material = useRef()
  const uniforms = useMemo(() => ({ uTime: { value: 0 } }), [])
  useFrame((state) => {
    material.current.uniforms.uTime.value = state.clock.elapsedTime
  })
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.05, 0]}>
      <planeGeometry args={[11, 11]} />
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={rippleVertex}
        fragmentShader={rippleFragment}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  )
}

export default function ParticleStory({ reducedMotion = false, onStage }) {
  const small = typeof window !== 'undefined' && window.innerWidth < 900
  return (
    <Canvas
      aria-hidden="true"
      dpr={[1, 2]}
      frameloop={reducedMotion ? 'demand' : 'always'}
      camera={{ position: [0, 2.1, 7.6], fov: 42 }}
      onCreated={({ camera }) => camera.lookAt(0, 0.7, 0)}
      gl={{ antialias: false, alpha: true, powerPreference: 'high-performance' }}
      style={{ background: 'transparent' }}
    >
      <Particles count={small ? 7000 : 16000} reducedMotion={reducedMotion} onStage={onStage} />
      <Ripples />
    </Canvas>
  )
}

import { Canvas, useFrame } from '@react-three/fiber'
import { Environment, Float, Lightformer, RoundedBox } from '@react-three/drei'
import { useRef } from 'react'

/*
  Floating steel-blue shapes for the hero areas: a medical cross, a capsule, a ring, a gem and a few
  spheres, drifting slowly. Lit by soft studio panels (built in code, no downloads), so they read as
  polished glass-metal against the navy page. Decoration only; hidden from assistive technology.
*/

function Material({ tint = '#9db8d8', roughness = 0.18 }) {
  return (
    <meshPhysicalMaterial
      color={tint}
      roughness={roughness}
      metalness={0.12}
      clearcoat={1}
      clearcoatRoughness={0.12}
      iridescence={0.25}
      iridescenceIOR={1.3}
      envMapIntensity={1.6}
    />
  )
}

function Cross(props) {
  return (
    <group {...props}>
      <RoundedBox args={[0.42, 1.3, 0.42]} radius={0.12} smoothness={6}>
        <Material tint="#dce7f4" roughness={0.12} />
      </RoundedBox>
      <RoundedBox args={[1.3, 0.42, 0.42]} radius={0.12} smoothness={6}>
        <Material tint="#dce7f4" roughness={0.12} />
      </RoundedBox>
    </group>
  )
}

function Capsule(props) {
  return (
    <group {...props}>
      <mesh position={[0, 0.32, 0]}>
        <capsuleGeometry args={[0.3, 0.64, 12, 32]} />
        <Material tint="#80a0c5" />
      </mesh>
    </group>
  )
}

function Ring(props) {
  return (
    <mesh {...props}>
      <torusGeometry args={[0.62, 0.17, 32, 96]} />
      <Material tint="#c3d5ea" roughness={0.1} />
    </mesh>
  )
}

function Gem(props) {
  return (
    <mesh {...props}>
      <icosahedronGeometry args={[0.55, 0]} />
      <Material tint="#a99bff" roughness={0.08} />
    </mesh>
  )
}

function Orb({ tint = '#9db8d8', size = 0.28, ...props }) {
  return (
    <mesh {...props}>
      <sphereGeometry args={[size, 48, 48]} />
      <Material tint={tint} roughness={0.05} />
    </mesh>
  )
}

/** The whole group leans slightly toward the pointer, so the scene feels alive without being busy. */
function Drift({ children }) {
  const group = useRef()
  useFrame((state) => {
    group.current.rotation.y += (state.pointer.x * 0.18 - group.current.rotation.y) * 0.03
    group.current.rotation.x += (-state.pointer.y * 0.1 - group.current.rotation.x) * 0.03
  })
  return <group ref={group}>{children}</group>
}

// Positions for the two layouts: around a centred picture (hero) or to the side of a banner.
const LAYOUTS = {
  hero: [
    ['cross', [1.9, 2.4, -0.8], 0.7],
    ['capsule', [2.05, -0.1, 0.4], 0.5],
    ['ring', [-2.15, -0.9, 0], 0.55],
    ['gem', [1.85, -2.45, -0.3], 0.5],
    ['orb', [-2.0, 0.9, 0.6], 0.35],
    ['orb', [-0.6, -3.0, -1], 0.35],
    ['orb', [2.3, 1.0, -1.6], 0.35],
  ],
  banner: [
    ['cross', [1.2, 0.5, 0], 0.9],
    ['ring', [3.1, -0.4, -0.6], 0.8],
    ['capsule', [-0.6, -0.9, -0.8], 0.75],
    ['gem', [2.6, 1.3, -1.2], 0.6],
    ['orb', [-1.3, 1.1, -1.4], 0.7],
    ['orb', [4.0, 0.9, -1.8], 0.6],
  ],
}

const SHAPES = { cross: Cross, capsule: Capsule, ring: Ring, gem: Gem, orb: Orb }

export default function FloatingShapes({ variant = 'hero', reducedMotion = false }) {
  const layout = LAYOUTS[variant] || LAYOUTS.hero
  return (
    <Canvas
      aria-hidden="true"
      dpr={[1, 1.75]}
      frameloop={reducedMotion ? 'demand' : 'always'}
      camera={{ position: [0, 0, 8], fov: 40 }}
      gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
      style={{ background: 'transparent' }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[3, 5, 4]} intensity={1.2} />
      <Drift>
        {layout.map(([kind, position, scale], index) => {
          const Shape = SHAPES[kind]
          return (
            <Float
              key={`${kind}-${index}`}
              speed={reducedMotion ? 0 : 1.1 + (index % 3) * 0.35}
              rotationIntensity={reducedMotion ? 0 : 0.9}
              floatIntensity={reducedMotion ? 0 : 1.2}
              floatingRange={[-0.15, 0.15]}
            >
              <Shape position={position} scale={scale} rotation={[0.3 * index, 0.5 * index, 0.2]} />
            </Float>
          )
        })}
      </Drift>
      {/* Soft studio panels: the reflections that make the shapes look polished. */}
      <Environment resolution={256} frames={1}>
        <Lightformer form="rect" intensity={3} position={[0, 4, 3]} scale={[8, 2, 1]} color="#ffffff" />
        <Lightformer form="rect" intensity={1.6} position={[-5, 0, 2]} rotation-y={Math.PI / 2} scale={[6, 4, 1]} color="#80a0c5" />
        <Lightformer form="rect" intensity={1.4} position={[5, -1, 2]} rotation-y={-Math.PI / 2} scale={[6, 4, 1]} color="#a99bff" />
        <Lightformer form="ring" intensity={2} position={[0, -3, 4]} scale={3} color="#dce7f4" />
      </Environment>
    </Canvas>
  )
}

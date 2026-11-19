import { Canvas, useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

/*
  A quiet, data-like picture of the federation: a globe drawn as fine dots, hospitals as points on
  its surface, and thin arcs to the coordinator. Small pulses run along the arcs: model updates in
  (blue) and the new global model out (violet). It illustrates one round; it is not data.
*/

const RADIUS = 2
const DOT = '#7f9ddb'
const NODE = '#2d59b5'
const ARC = '#7d9be0'
const IN = '#2d59b5'
const OUT = '#7c5cff'

function fibonacciSphere(count, radius) {
  const positions = new Float32Array(count * 3)
  const golden = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (i / (count - 1)) * 2
    const r = Math.sqrt(1 - y * y)
    const theta = golden * i
    positions.set([Math.cos(theta) * r * radius, y * radius, Math.sin(theta) * r * radius], i * 3)
  }
  return positions
}

function onSphere(lat, lon, radius = RADIUS) {
  const phi = THREE.MathUtils.degToRad(90 - lat)
  const theta = THREE.MathUtils.degToRad(lon)
  return new THREE.Vector3(radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi), radius * Math.sin(phi) * Math.sin(theta))
}

function arc(from, to) {
  const mid = from.clone().add(to).multiplyScalar(0.5)
  const lift = 1 + from.distanceTo(to) / (RADIUS * 2.2)
  mid.normalize().multiplyScalar(RADIUS * lift)
  return new THREE.QuadraticBezierCurve3(from, mid, to)
}

function Pulse({ curve, color, offset, speed, reverse }) {
  const mesh = useRef()
  const point = useMemo(() => new THREE.Vector3(), [])
  useFrame((state) => {
    let t = (state.clock.elapsedTime * speed + offset) % 1
    if (reverse) t = 1 - t
    curve.getPoint(t, point)
    mesh.current.position.copy(point)
    mesh.current.material.opacity = Math.sin(Math.PI * t)
  })
  return (
    <mesh ref={mesh}>
      <sphereGeometry args={[0.03, 12, 12]} />
      <meshBasicMaterial color={color} transparent />
    </mesh>
  )
}

function Node({ position, color, size = 0.045, halo = true }) {
  const ring = useRef()
  const quaternion = useMemo(
    () => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), position.clone().normalize()),
    [position],
  )
  useFrame((state) => {
    if (!ring.current) return
    const t = (state.clock.elapsedTime * 0.6 + Math.abs(position.x)) % 1
    ring.current.scale.setScalar(1 + t * 1.2)
    ring.current.material.opacity = 0.35 * (1 - t)
  })
  return (
    <group position={position.toArray()}>
      <mesh>
        <sphereGeometry args={[size, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
      {halo ? (
        <mesh ref={ring} quaternion={quaternion}>
          <ringGeometry args={[size * 1.5, size * 1.75, 32]} />
          <meshBasicMaterial color={color} transparent side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ) : null}
    </group>
  )
}

// Hospitals spread over the visible face of the globe; the coordinator sits among them.
const SPOTS = [
  [44, 20], [-22, 28], [40, 88], [-26, 80], [8, 0], [10, 104], [52, 55], [-40, 52],
]
const COORDINATOR = [14, 52]

function Globe({ hospitals, active }) {
  const root = useRef()
  const dots = useMemo(() => fibonacciSphere(2600, RADIUS), [])
  const hub = useMemo(() => onSphere(...COORDINATOR, RADIUS * 1.01), [])
  const links = useMemo(
    () =>
      hospitals.map((hospital, index) => {
        const position = onSphere(...SPOTS[index % SPOTS.length], RADIUS * 1.01)
        const curve = arc(position, hub)
        return { ...hospital, position, curve, points: curve.getPoints(48) }
      }),
    [hospitals, hub],
  )

  useFrame((state, delta) => {
    root.current.rotation.y += delta * 0.035
    const targetX = 0.18 + state.pointer.y * 0.06
    root.current.rotation.x += (targetX - root.current.rotation.x) * 0.04
  })

  const speed = active ? 0.5 : 0.28
  return (
    <group ref={root} rotation={[0.18, -0.9, 0]}>
      <points renderOrder={1}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[dots, 3]} />
        </bufferGeometry>
        <pointsMaterial color={DOT} size={0.03} sizeAttenuation transparent opacity={0.85} depthWrite={false} />
      </points>
      {/* A faint body so dots on the far side read as further away. */}
      <mesh renderOrder={0}>
        <sphereGeometry args={[RADIUS * 0.985, 48, 48]} />
        <meshBasicMaterial color="#f3f6fd" transparent opacity={0.6} depthWrite />
      </mesh>
      <Node position={hub} color={OUT} size={0.06} />
      {links.map((link, index) => {
        const online = link.online !== false
        return (
          <group key={link.id ?? index}>
            <Line points={link.points} color={online ? ARC : '#c9d1de'} lineWidth={1} transparent opacity={online ? 0.75 : 0.5} />
            <Node position={link.position} color={online ? NODE : '#a5afbf'} halo={online} />
            {online ? (
              <>
                <Pulse curve={link.curve} color={IN} offset={index * 0.23} speed={speed} />
                <Pulse curve={link.curve} color={OUT} offset={index * 0.23 + 0.5} speed={speed} reverse />
              </>
            ) : null}
          </group>
        )
      })}
    </group>
  )
}

export default function FederationScene({ hospitals, active = false, reducedMotion = false, cameraZ = 7 }) {
  const list = hospitals?.length ? hospitals : [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }]
  return (
    <Canvas
      aria-hidden="true"
      flat
      dpr={[1, 2]}
      frameloop={reducedMotion ? 'demand' : 'always'}
      camera={{ position: [0, 0, cameraZ], fov: 36 }}
      gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
      style={{ background: 'transparent' }}
    >
      <Globe hospitals={list} active={active} />
    </Canvas>
  )
}

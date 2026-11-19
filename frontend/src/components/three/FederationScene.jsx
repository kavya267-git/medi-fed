import { Canvas, useFrame } from '@react-three/fiber'
import { Line, RoundedBox } from '@react-three/drei'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

/*
  The federation in 3D: a glowing coordinator core in the middle, hospitals on a ring around it.
  Blue particles travel from each hospital to the core (model updates), violet ones travel back
  (the new global model). Nothing here is data; it illustrates what one round does.
*/

const BLUE = '#2d59b5'
const VIOLET = '#7c5cff'
const SKY = '#8fb4f5'

function arcBetween(from, to, lift) {
  const mid = from.clone().add(to).multiplyScalar(0.5)
  mid.y += lift
  return new THREE.QuadraticBezierCurve3(from, mid, to)
}

function Core({ active }) {
  const shell = useRef()
  const ringA = useRef()
  const ringB = useRef()
  const glow = useRef()
  useFrame((state, delta) => {
    const speed = active ? 2.2 : 1
    shell.current.rotation.y += delta * 0.25 * speed
    shell.current.rotation.x += delta * 0.1 * speed
    ringA.current.rotation.z += delta * 0.4 * speed
    ringB.current.rotation.z -= delta * 0.3 * speed
    const pulse = 1 + Math.sin(state.clock.elapsedTime * (active ? 4 : 2)) * 0.04
    glow.current.scale.setScalar(pulse)
  })
  return (
    <group position={[0, 0.35, 0]}>
      <mesh ref={glow}>
        <sphereGeometry args={[0.62, 48, 48]} />
        <meshStandardMaterial color={BLUE} emissive={BLUE} emissiveIntensity={0.9} roughness={0.25} metalness={0.1} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.95, 32, 32]} />
        <meshBasicMaterial color={SKY} transparent opacity={0.16} depthWrite={false} />
      </mesh>
      <mesh ref={shell}>
        <icosahedronGeometry args={[1.05, 1]} />
        <meshBasicMaterial color={VIOLET} wireframe transparent opacity={0.45} />
      </mesh>
      <mesh ref={ringA} rotation={[Math.PI / 2.3, 0, 0]}>
        <torusGeometry args={[1.35, 0.012, 8, 96]} />
        <meshBasicMaterial color={BLUE} transparent opacity={0.55} />
      </mesh>
      <mesh ref={ringB} rotation={[Math.PI / 1.7, 0.4, 0]}>
        <torusGeometry args={[1.55, 0.01, 8, 96]} />
        <meshBasicMaterial color={VIOLET} transparent opacity={0.4} />
      </mesh>
      {/* Server base under the core, like the illustration. */}
      <mesh position={[0, -1.05, 0]}>
        <cylinderGeometry args={[1.0, 1.12, 0.1, 48]} />
        <meshStandardMaterial color="#d6e4fb" roughness={0.7} />
      </mesh>
    </group>
  )
}

function Hospital({ position, online, index }) {
  const group = useRef()
  useFrame((state) => {
    group.current.position.y = position.y + Math.sin(state.clock.elapsedTime * 1.2 + index) * 0.05
  })
  return (
    <group ref={group} position={position.toArray()}>
      <mesh position={[0, -0.32, 0]}>
        <cylinderGeometry args={[0.5, 0.54, 0.08, 40]} />
        <meshStandardMaterial color={online ? '#cfe0fb' : '#e4e8ee'} roughness={0.8} />
      </mesh>
      <RoundedBox args={[0.62, 0.5, 0.48]} radius={0.06} smoothness={3}>
        <meshStandardMaterial color="#ffffff" roughness={0.45} />
      </RoundedBox>
      <RoundedBox args={[0.36, 0.26, 0.34]} radius={0.04} smoothness={3} position={[0, 0.36, 0]}>
        <meshStandardMaterial color={online ? '#eef4ff' : '#f1f3f6'} roughness={0.45} />
      </RoundedBox>
      {/* Red cross on the front face. */}
      <group position={[0, 0.37, 0.175]}>
        <mesh>
          <boxGeometry args={[0.16, 0.05, 0.01]} />
          <meshBasicMaterial color={online ? '#e5484d' : '#b8bfc9'} />
        </mesh>
        <mesh>
          <boxGeometry args={[0.05, 0.16, 0.01]} />
          <meshBasicMaterial color={online ? '#e5484d' : '#b8bfc9'} />
        </mesh>
      </group>
      {/* Windows. */}
      {[-0.16, 0, 0.16].map((x) => (
        <mesh key={x} position={[x, 0.02, 0.245]}>
          <boxGeometry args={[0.09, 0.12, 0.01]} />
          <meshStandardMaterial color="#9fbdf0" emissive="#5b8de8" emissiveIntensity={online ? 0.35 : 0} />
        </mesh>
      ))}
    </group>
  )
}

function Packet({ curve, color, offset, speed, reverse }) {
  const mesh = useRef()
  const point = useMemo(() => new THREE.Vector3(), [])
  useFrame((state) => {
    let t = (state.clock.elapsedTime * speed + offset) % 1
    if (reverse) t = 1 - t
    curve.getPoint(t, point)
    mesh.current.position.copy(point)
    const fade = Math.sin(Math.PI * t)
    mesh.current.scale.setScalar(0.6 + fade * 0.6)
  })
  return (
    <mesh ref={mesh}>
      <sphereGeometry args={[0.075, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  )
}

function Federation({ hospitals, active, pointerTilt }) {
  const root = useRef()
  const nodes = useMemo(() => {
    const count = Math.max(1, hospitals.length)
    return hospitals.map((hospital, index) => {
      const angle = (index / count) * Math.PI * 2 + Math.PI / 4
      const position = new THREE.Vector3(Math.cos(angle) * 2.9, -0.35, Math.sin(angle) * 2.0)
      const curve = arcBetween(position.clone().add(new THREE.Vector3(0, 0.3, 0)), new THREE.Vector3(0, 0.35, 0), 1.1)
      return { ...hospital, position, curve, points: curve.getPoints(40) }
    })
  }, [hospitals])

  useFrame((state, delta) => {
    root.current.rotation.y += delta * (active ? 0.12 : 0.06)
    if (pointerTilt) {
      const targetX = state.pointer.y * 0.12
      root.current.rotation.x += (targetX - root.current.rotation.x) * 0.05
    }
  })

  const speed = active ? 0.42 : 0.22
  return (
    <group ref={root}>
      <Core active={active} />
      {nodes.map((node, index) => (
        <group key={node.id ?? index}>
          <Hospital position={node.position} online={node.online !== false} index={index} />
          <Line points={node.points} color={node.online === false ? '#c5ccd8' : '#7fa3e8'} lineWidth={1.2} dashed dashSize={0.12} gapSize={0.09} transparent opacity={0.8} />
          {node.online === false ? null : (
            <>
              <Packet curve={node.curve} color={BLUE} offset={index * 0.17} speed={speed} />
              <Packet curve={node.curve} color={VIOLET} offset={index * 0.17 + 0.5} speed={speed} reverse />
            </>
          )}
        </group>
      ))}
    </group>
  )
}

export default function FederationScene({ hospitals, active = false, reducedMotion = false, pointerTilt = true, cameraZ = 7.1 }) {
  const list = hospitals?.length ? hospitals : [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }]
  return (
    <Canvas
      aria-hidden="true"
      dpr={[1, 2]}
      frameloop={reducedMotion ? 'demand' : 'always'}
      flat
      camera={{ position: [0, 3.4, cameraZ], fov: 38 }}
      onCreated={({ camera }) => camera.lookAt(0, 0, 0)}
      gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
      style={{ background: 'transparent' }}
    >
      <ambientLight intensity={1.6} />
      <directionalLight position={[4, 8, 5]} intensity={1.8} color="#ffffff" />
      <directionalLight position={[-6, 3, -4]} intensity={0.5} color={SKY} />
      <pointLight position={[0, 0.4, 0]} intensity={6} distance={5} color={BLUE} />
      <Federation hospitals={list} active={active} pointerTilt={pointerTilt && !reducedMotion} />
    </Canvas>
  )
}

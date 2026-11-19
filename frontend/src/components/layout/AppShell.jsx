import { LogOut, Menu, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { motion } from 'motion/react'
import { useAuth } from '../../context/AuthContext'
import { useSystem } from '../../context/SystemContext'
import { ROLE_LABELS } from '../../utils/constants'
import { classNames } from '../../utils/format'
import { SimulatedTag } from '../ui/Badge'
import { Logo } from './Logo'
import { api } from '../../services/endpoints'
import { navItemsFor } from './navigation'

/** Pending hospital registrations, for the administrator's nav badge. Refreshed every 30 s and after a review. */
function usePendingRegistrations(user) {
  const [count, setCount] = useState(0)
  const load = useCallback(() => {
    if (user?.role !== 'admin') return
    // A missing count must never break the page: the badge is a hint, the page itself has the list.
    Promise.resolve()
      .then(() => api.pendingRegistrations())
      .then((result) => setCount(Number(result?.pending) || 0))
      .catch(() => {})
  }, [user?.role])
  useEffect(() => {
    load()
    if (user?.role !== 'admin') return undefined
    const timer = setInterval(load, 30000)
    window.addEventListener('medifed:registrations-changed', load)
    return () => {
      clearInterval(timer)
      window.removeEventListener('medifed:registrations-changed', load)
    }
  }, [load, user?.role])
  return count
}

function Navigation({ onNavigate, layoutId = 'nav-active' }) {
  const { user } = useAuth()
  const pending = usePendingRegistrations(user)
  return (
    <nav aria-label="Main" className="flex flex-col gap-0.5 px-3">
      {navItemsFor(user).map(({ to, label, icon: Icon, end, badge }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) =>
            classNames(
              'relative flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-colors',
              isActive ? 'text-white' : 'text-muted hover:bg-brand-50 hover:text-brand-800',
            )
          }
        >
          {({ isActive }) => (
            <>
              {isActive ? (
                <motion.span
                  layoutId={layoutId}
                  className="absolute inset-0 rounded-xl bg-linear-to-r from-brand-600 to-violet shadow-[0_8px_20px_-10px_rgb(45_89_181/0.7)]"
                  transition={{ type: 'spring', stiffness: 420, damping: 36 }}
                  aria-hidden="true"
                />
              ) : null}
              <Icon className="relative h-4 w-4 shrink-0" aria-hidden="true" />
              <span className="relative">{label}</span>
          {badge === 'pendingRegistrations' && pending > 0 ? (
            <span
              className="relative ml-auto rounded-full bg-warn px-2 py-0.5 text-xs font-semibold text-white"
              aria-label={`${pending} pending`}
            >
              {pending}
            </span>
          ) : null}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  )
}

function ModeNotice() {
  const { config } = useSystem()
  if (!config?.demo_mode) return null
  return (
    <div className="mx-3 rounded-xl border border-brand-100 bg-brand-50/70 px-3 py-3">
      <p className="text-xs font-semibold text-brand-900">Demo mode</p>
      <p className="mt-1 text-xs leading-relaxed text-muted">
        Hospitals and data are simulated. Training and metrics are computed for real on that data.
      </p>
    </div>
  )
}

export function AppShell() {
  const { user, signOut } = useAuth()
  const { config } = useSystem()
  const location = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => setMenuOpen(false), [location.pathname])

  return (
    <div className="min-h-screen lg:flex">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>

      {/* Sidebar (desktop) */}
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-line/70 bg-white/75 backdrop-blur-xl lg:flex">
        <div className="px-5 py-5">
          <Logo />
        </div>
        <div className="flex-1 overflow-y-auto pb-4">
          <Navigation />
        </div>
        <div className="pb-4">
          <ModeNotice />
        </div>
      </aside>

      {/* Drawer (mobile) */}
      {menuOpen ? (
        <div className="fixed inset-0 z-[1200] lg:hidden">
          <button type="button" aria-label="Close menu" className="absolute inset-0 bg-brand-950/50" onClick={() => setMenuOpen(false)} />
          <div className="relative flex h-full w-72 max-w-[85%] flex-col bg-surface pb-4">
            <div className="flex items-center justify-between px-5 py-5">
              <Logo />
              <button type="button" aria-label="Close menu" onClick={() => setMenuOpen(false)} className="rounded p-1 text-muted">
                <X className="h-5 w-5" aria-hidden="true" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              <Navigation onNavigate={() => setMenuOpen(false)} layoutId="nav-active-mobile" />
            </div>
            <ModeNotice />
          </div>
        </div>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-[1000] flex h-14 items-center justify-between gap-3 border-b border-line bg-white/70 px-4 backdrop-blur-xl lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <button
              type="button"
              aria-label="Open menu"
              onClick={() => setMenuOpen(true)}
              className="rounded p-1.5 text-muted hover:bg-canvas lg:hidden"
            >
              <Menu className="h-5 w-5" aria-hidden="true" />
            </button>
            {config?.demo_mode ? <SimulatedTag label={config.simulated_label} /> : null}
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden text-right leading-tight sm:block">
              <p className="max-w-56 truncate text-sm font-medium text-ink">{user?.display_name || user?.email}</p>
              <p className="text-xs text-muted">{ROLE_LABELS[user?.role] || user?.role}</p>
            </div>
            <button
              type="button"
              onClick={signOut}
              className="inline-flex h-9 items-center gap-2 rounded-xl border border-line-strong bg-white/80 px-3 text-sm font-medium text-ink hover:bg-brand-50"
            >
              <LogOut className="h-4 w-4" aria-hidden="true" />
              Sign out
            </button>
          </div>
        </header>
        <main id="main" className="mx-auto w-full max-w-[1320px] flex-1 px-4 py-7 lg:px-8">
          {/* Each page arrives with one short rise; the navigation pill slides to show where you went. */}
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          >
            <Outlet />
          </motion.div>
        </main>
      </div>
    </div>
  )
}

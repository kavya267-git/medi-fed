import { Moon, Sun } from 'lucide-react'
import { useTheme } from '../../hooks/useTheme'
import { classNames } from '../../utils/format'

/** Switches between the light and the dark theme. */
export function ThemeToggle({ className = '' }) {
  const { theme, toggle } = useTheme()
  const next = theme === 'light' ? 'dark' : 'light'
  const Icon = theme === 'light' ? Moon : Sun
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className={classNames(
        'inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-white/20 text-ink hover:bg-white/10',
        className,
      )}
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
    </button>
  )
}

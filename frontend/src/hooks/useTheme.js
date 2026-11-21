import { useCallback, useEffect, useState } from 'react'

/** Light or dark, kept per browser. index.html applies the saved choice before the first paint. */
const KEY = 'medifed-theme'
const EVENT = 'medifed-theme-change'

export function currentTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem(KEY, theme)
  } catch {
    // Storage can be blocked (private window); the choice then lasts until the page is reloaded.
  }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: theme }))
}

export function useTheme() {
  const [theme, setTheme] = useState(currentTheme)
  useEffect(() => {
    const sync = () => setTheme(currentTheme())
    window.addEventListener(EVENT, sync)
    return () => window.removeEventListener(EVENT, sync)
  }, [])
  const toggle = useCallback(() => applyTheme(currentTheme() === 'light' ? 'dark' : 'light'), [])
  return { theme, toggle }
}

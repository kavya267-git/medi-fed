import { LoaderCircle } from 'lucide-react'
import { classNames } from '../../utils/format'

const VARIANTS = {
  primary: 'bg-cta text-cta-ink hover:bg-brand-900 disabled:bg-white/40',
  secondary: 'border border-white/20 bg-white/5 text-ink hover:bg-white/10 disabled:text-faint',
  danger: 'border border-alert/40 bg-transparent text-alert hover:bg-alert-soft disabled:text-faint',
  ghost: 'text-brand-700 hover:bg-white/10 disabled:text-faint',
}

const SIZES = { md: 'h-10 px-5 text-sm', sm: 'h-8 px-3.5 text-[13px]' }

export function Button({
  children,
  variant = 'secondary',
  size = 'md',
  loading = false,
  icon: Icon,
  className = '',
  type = 'button',
  disabled,
  ...rest
}) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={classNames(
        'inline-flex items-center justify-center gap-2 rounded-full font-semibold transition-[color,background-color,transform] active:scale-[0.98] disabled:active:scale-100',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {loading ? (
        <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
      ) : Icon ? (
        <Icon className="h-4 w-4" aria-hidden="true" />
      ) : null}
      {children}
    </button>
  )
}

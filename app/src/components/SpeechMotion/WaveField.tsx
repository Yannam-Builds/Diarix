import { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils/cn';

type WaveVariant = 'dashboard' | 'onboarding';

interface WaveLine {
  amplitude: number;
  wavelength: number;
  speed: number;
  y: number;
  phase: number;
  alpha: number;
}

interface WaveFieldProps {
  variant?: WaveVariant;
  className?: string;
}

const DASHBOARD_LINES: WaveLine[] = [
  { amplitude: 14, wavelength: 178, speed: 78, y: 70, phase: 0.1, alpha: 0.42 },
  { amplitude: 11, wavelength: 148, speed: 92, y: 74, phase: 1.2, alpha: 0.34 },
  { amplitude: 9, wavelength: 128, speed: 66, y: 78, phase: 2.1, alpha: 0.28 },
  { amplitude: 7, wavelength: 204, speed: 84, y: 82, phase: 3, alpha: 0.22 },
  { amplitude: 5, wavelength: 112, speed: 58, y: 86, phase: 4, alpha: 0.16 },
];

const ONBOARDING_LINES: WaveLine[] = Array.from({ length: 15 }, (_, index) => {
  const position = index / 14;
  const center = 1 - Math.abs(position - 0.5) * 2;
  return {
    amplitude: 12 + center * 42 + ((index * 17) % 9),
    wavelength: 112 + ((index * 47) % 176),
    speed: 24 + ((index * 19) % 44),
    y: 10 + position * 130,
    phase: (index * 1.31) % (Math.PI * 2),
    alpha: 0.045 + center * 0.18,
  };
});

const VIEWBOX_WIDTH = 640;
const VIEWBOX_HEIGHT = 150;
export function WaveField({ variant = 'dashboard', className }: WaveFieldProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!host || !canvas || !context) return;

    const lines = variant === 'onboarding' ? ONBOARDING_LINES : DASHBOARD_LINES;
    const points = variant === 'onboarding' ? 128 : 150;
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let reducedMotion = motionQuery.matches;
    let isIntersecting = true;
    let documentVisible = document.visibilityState === 'visible';
    let width = 1;
    let height = 1;
    let frame: number | null = null;

    const shouldAnimate = () => !reducedMotion && isIntersecting && documentVisible;

    const paint = (now: number) => {
      context.clearRect(0, 0, width, height);
      context.lineWidth = variant === 'onboarding' ? 1.1 : 1.15;
      context.lineCap = 'round';
      context.lineJoin = 'round';

      const accent = getComputedStyle(canvas).getPropertyValue('--accent').trim() || '43 55% 58%';
      const time = now * 0.001;
      const scaleX = width / VIEWBOX_WIDTH;
      const scaleY = height / VIEWBOX_HEIGHT;

      for (const line of lines) {
        const travel = reducedMotion ? 0 : time * line.speed;
        const breathe = reducedMotion ? 0 : Math.sin(time * 0.8 + line.phase) * 1.5;
        context.beginPath();

        for (let point = 0; point <= points; point += 1) {
          const viewX = (VIEWBOX_WIDTH / points) * point;
          const normalized = point / points;
          const envelope =
            variant === 'dashboard'
              ? 0.28 + 0.72 * Math.sin(Math.PI * normalized)
              : Math.sin(Math.PI * normalized);
          const primary =
            Math.sin(((viewX - travel) / line.wavelength) * Math.PI * 2 + line.phase) *
            line.amplitude *
            envelope;
          const harmonic =
            variant === 'dashboard'
              ? Math.sin(
                  ((viewX - travel * 0.72) / (line.wavelength * 0.58)) * Math.PI * 2 +
                    line.phase * 0.6,
                ) *
                line.amplitude *
                0.16
              : 0;
          const x = viewX * scaleX;
          const y = (line.y + primary + harmonic + breathe) * scaleY;

          if (point === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }

        context.strokeStyle = 'hsl(' + accent + ' / ' + line.alpha + ')';
        context.stroke();
      }
    };

    const render = (now: number) => {
      frame = null;
      paint(now);
      if (shouldAnimate()) frame = requestAnimationFrame(render);
    };

    const requestFrame = () => {
      if (frame === null && shouldAnimate()) frame = requestAnimationFrame(render);
    };

    const pause = () => {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
    };

    const resize = () => {
      const rect = host.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.max(1, Math.floor(width * ratio));
      canvas.height = Math.max(1, Math.floor(height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      paint(performance.now());
      requestFrame();
    };

    const handleVisibility = () => {
      documentVisible = document.visibilityState === 'visible';
      if (shouldAnimate()) requestFrame();
      else pause();
    };

    const handleMotionPreference = (event: MediaQueryListEvent) => {
      reducedMotion = event.matches;
      pause();
      paint(performance.now());
      requestFrame();
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);

    const intersectionObserver = new IntersectionObserver(
      ([entry]) => {
        isIntersecting = Boolean(entry?.isIntersecting);
        if (shouldAnimate()) requestFrame();
        else pause();
      },
      { threshold: 0.05 },
    );
    intersectionObserver.observe(host);

    document.addEventListener('visibilitychange', handleVisibility);
    motionQuery.addEventListener('change', handleMotionPreference);
    resize();

    return () => {
      pause();
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      document.removeEventListener('visibilitychange', handleVisibility);
      motionQuery.removeEventListener('change', handleMotionPreference);
    };
  }, [variant]);

  return (
    <div
      ref={hostRef}
      className={cn(
        'pointer-events-none absolute -left-[7%] -right-[7%] inset-y-0 overflow-visible',
        className,
      )}
      style={{
        maskImage: 'linear-gradient(90deg, transparent 0%, black 10%, black 90%, transparent 100%)',
        WebkitMaskImage:
          'linear-gradient(90deg, transparent 0%, black 10%, black 90%, transparent 100%)',
      }}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  );
}

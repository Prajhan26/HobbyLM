import type { ElementType, ReactNode } from "react";

export function SignalHeading({ as: Tag = "h2", children, className = "section-title", id }: { as?: ElementType; children: ReactNode; className?: string; id?: string }) {
  return <Tag className={className} id={id}>{children}<span className="signal-square" aria-hidden="true" /></Tag>;
}

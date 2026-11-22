"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { Icon } from "./Icon";

/**
 * Bottom sheet built on the native <dialog> element: showModal() gives focus trapping,
 * Escape to close, an inert background and focus return to the trigger.
 */
export function Sheet({
  open,
  onClose,
  title,
  children,
  footer,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      aria-labelledby="sheet-title"
      onClose={onClose}
      onClick={(event) => {
        if (event.target === ref.current) onClose(); // click on the backdrop
      }}
      className="sheet m-0 mt-auto w-full max-w-none rounded-t-[16px] border border-line-strong bg-raised p-0 text-fg shadow-[var(--shadow-raised)] backdrop:bg-black/55"
    >
      <div className="flex items-center justify-between border-b border-line px-5 py-3">
        <h2 id="sheet-title" className="h3">
          {title}
        </h2>
        <button type="button" onClick={onClose} className="btn btn-ghost -mr-3 px-3" aria-label="Close">
          <Icon name="cross" />
        </button>
      </div>
      <div className="max-h-[70vh] overflow-y-auto px-5 py-5">{children}</div>
      {footer && <div className="border-t border-line px-5 py-3">{footer}</div>}
    </dialog>
  );
}

"use client";

import { useSyncExternalStore } from "react";
import { Icon, type IconName } from "../ui/Icon";

type Choice = "system" | "dark" | "light";
const KEY = "recalliq-theme";
const OPTIONS: { value: Choice; label: string; icon: IconName }[] = [
  { value: "system", label: "System", icon: "monitor" },
  { value: "dark", label: "Dark", icon: "moon" },
  { value: "light", label: "Light", icon: "sun" },
];

/**
 * Runs before paint (inlined in <head>) so the page never flashes the wrong theme.
 * Dark is the brand default; "system" follows the OS once chosen.
 */
export const THEME_SCRIPT =
  "(function(){try{var c=localStorage.getItem('" + KEY + "');var d=c==='light'||c==='dark'?c:" +
  "c==='system'?(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'):'dark';" +
  "document.documentElement.dataset.theme=d;}catch(e){document.documentElement.dataset.theme='dark';}})();";

const listeners = new Set<() => void>();

function read(): Choice {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" || value === "system" ? value : "dark";
  } catch {
    return "dark";
  }
}

function apply(choice: Choice) {
  try {
    localStorage.setItem(KEY, choice);
  } catch {
    /* storage blocked: still switch for this page view */
  }
  const resolved =
    choice === "system" ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark") : choice;
  document.documentElement.dataset.theme = resolved;
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function ThemeToggle() {
  const choice = useSyncExternalStore(subscribe, read, () => "dark" as Choice);
  return (
    <div role="radiogroup" aria-label="Colour theme" className="inline-flex rounded-full border border-line p-0.5">
      {OPTIONS.map((option) => {
        const checked = choice === option.value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={checked}
            onClick={() => apply(option.value)}
            onKeyDown={(event) => {
              const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
              if (!step) return;
              event.preventDefault();
              const index = (OPTIONS.findIndex((o) => o.value === choice) + step + OPTIONS.length) % OPTIONS.length;
              apply(OPTIONS[index].value);
              (event.currentTarget.parentElement?.children[index] as HTMLElement | undefined)?.focus();
            }}
            tabIndex={checked ? 0 : -1}
            className={
              "inline-flex h-11 w-11 items-center justify-center rounded-full transition-colors " +
              (checked ? "bg-raised text-accent" : "text-fg-3 hover:text-fg")
            }
          >
            <Icon name={option.icon} size={16} label={option.label} />
          </button>
        );
      })}
    </div>
  );
}

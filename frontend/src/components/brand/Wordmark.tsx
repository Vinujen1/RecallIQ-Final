// TODO(logo): TEMPORARY typographic wordmark. The selected RecallIQ logo asset is not in the
// repository yet. When it arrives, put the SVGs in public/brand/ and change only this
// component; every page renders the brand through <Wordmark />.
export function Wordmark({ size = "md" }: { size?: "md" | "lg" }) {
  return (
    <span
      data-temporary-logo="true"
      className={
        "font-display font-semibold tracking-[-0.03em] leading-none " +
        (size === "lg" ? "text-[1.75rem]" : "text-[1.3rem]")
      }
    >
      <span className="text-fg">Recall</span>
      <span className="font-bold text-accent">IQ</span>
    </span>
  );
}

export interface ReadonlyMediaCompareProps {
  readonly leftTitle: string
  readonly rightTitle: string
  readonly leftSrc: string | null
  readonly rightSrc: string | null
  readonly kind: 'image' | 'video'
}

export function MediaCompare({ leftTitle, rightTitle, leftSrc, rightSrc, kind }: ReadonlyMediaCompareProps) {
  return (
    <div className="compare-grid">
      <section className="media-panel">
        <div className="media-heading">{leftTitle}</div>
        {leftSrc ? (
          kind === 'image' ? (
            <img className="media-fit" src={leftSrc} alt={leftTitle} />
          ) : (
            <video className="media-fit" src={leftSrc} controls playsInline />
          )
        ) : (
          <div className="media-placeholder">No media yet</div>
        )}
      </section>
      <section className="media-panel">
        <div className="media-heading">{rightTitle}</div>
        {rightSrc ? (
          kind === 'image' ? (
            <img className="media-fit" src={rightSrc} alt={rightTitle} />
          ) : (
            <video className="media-fit" src={rightSrc} controls playsInline />
          )
        ) : (
          <div className="media-placeholder">No media yet</div>
        )}
      </section>
    </div>
  )
}

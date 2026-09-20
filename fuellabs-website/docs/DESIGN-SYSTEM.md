# fuellabs. design system

This document is the visual source of truth. The live token values are in `styles/tokens.css`, and the rendered reference is available at `/style-guide`.

## Locked brand rules

- Background: Paper `#F2EFE8`
- Text: Ink `#11110F`
- Signal: Lime `#B7F34A`
- Secondary text: Muted `#66665F`
- Dividers: Line `#D8D4C9`
- Wordmark: black `fuel`, muted `labs`, lime square
- Page titles and major section headings end with a proportional lime square.
- Do not publish a location suffix with the brand.

## Typography

- Editorial display, headings, intros, body and navigation: General Sans 400/500
- Wordmark: IBM Plex Sans 500
- Technical labels, status and code: IBM Plex Mono 400/500

Use the semantic CSS classes and tokens (`display`, `section-title`, `intro-copy`, `body-copy`, `eyebrow`, `caption`). Page-specific font sizes require a written reason and screenshot review.

General Sans is currently loaded from Fontshare. Before public production, acquire and self-host the permitted webfont files to remove a third-party runtime dependency.

## Layout

- Maximum container: 1360px
- Fluid gutters: 24–80px
- Section padding: 128px desktop, 72px mobile
- Body copy: maximum 72 characters
- Intro copy: maximum 42 characters
- Primary responsive breakpoint: 800px

## Motion

Motion supports hierarchy. The home introduction is 1 second appearance, 0.8 second hold, and 1 second curtain rise. Every motion has a `prefers-reduced-motion` alternative.

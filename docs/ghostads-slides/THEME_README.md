# Ghost Ads Slides - Modern Theme Upgrade

**Date:** October 25, 2025
**Theme:** Gotham (keynote-ready configuration)
**Backup:** `slides_backup_20251025_031514/`

## What Changed

### 1. New Theme System
- **Before:** Basic Metropolis with manual color definitions
- **After:** Custom `ghostads-theme.sty` package based on Gotham theme
- **Benefits:** Keynote-ready styling, progress bars, better typography, icon bullets

### 2. Modern Color Palette
Replaced academic purple/blue/orange with corporate colors:

```latex
BrandPrimary   → #0F62FE (deep brand blue)
BrandSecondary → #5B8DEE (lighter blue)
BrandAccent    → #FF6F3D (energetic orange-red)
GhostPurple    → #7C3AED (for ghost-specific elements)
SuccessGreen   → #059669 (modern green)
ErrorRed       → #DC2626 (error/problem)
WarningOrange  → #EA580C (warning/alert)
```

### 3. New Features Added
- **Progress bar** in footer showing presentation progress
- **Icon-based bullets** using FontAwesome (checkmarks, arrows, etc.)
- **Standout frames** for high-impact slides (full-screen with brand background)
- **Better typography** with Fira Sans (clean, modern, readable)
- **Tighter margins** for more visual canvas
- **Rounded corners** on all boxes and diagrams (4-6pt radius)
- **Custom boxes:** keybox, insightbox, warningbox, successbox, ghostbox
- **Ghost icon** (`\ghostmark`) integrated throughout

### 4. Standout Frames Added
Three key slides now use standout frames for maximum impact:
1. **Part 1:** "Industry Leaders Use Ghost Ads"
2. **Part 3:** "Ghost Ads" (the big reveal)
3. **Part 4:** "iROAS Example"

## How to Use

### Basic Compilation
```bash
cd ghostads/slides
pdflatex main.tex
pdflatex main.tex  # Run twice for proper references
```

### Customizing Colors
Edit `ghostads-theme.sty` lines 35-46 to change brand colors:

```latex
\definecolor{BrandPrimary}{HTML}{YOUR_HEX}
\definecolor{BrandAccent}{HTML}{YOUR_HEX}
```

### Dark Mode
Uncomment in `main.tex` after `\usepackage{ghostads-theme}`:

```latex
\ghostadsdarkmode
```

### Changing Fonts
**For XeLaTeX/LuaLaTeX** (to use Inter or custom fonts):

In `ghostads-theme.sty`, replace lines 24-27 with:
```latex
\RequirePackage{fontspec}
\setsansfont{Inter}[Scale=0.95, Weight=300, BoldFont={* SemiBold}]
\renewcommand*\familydefault{\sfdefault}
```

### Progress Bar Options
In `ghostads-theme.sty` line 19, you can change:
- `progressbar=foot` → Shows progress bar in footer
- `progressbar=head` → Shows in header
- `progressbar=none` → Removes progress bar

### Section Pages
To add section dividers, insert in your content files:
```latex
\section{Section Name}
```

This will create an automatic section page with progress indicator.

## File Structure

```
slides/
├── ghostads-theme.sty        ← Custom theme (main config)
├── main.tex                  ← Master document
├── part1_problem.tex         ← Updated with standout
├── part2_traditional.tex     ← Updated colors
├── part3_ghostads.tex        ← Updated with standout
├── part4_implementation.tex  ← Updated with icons + standout
├── part5_extensions.tex      ← Updated colors
└── backup_slides.tex         ← Unchanged
```

## Quick Customization Examples

### Add a Standout Frame
```latex
\begin{frame}[standout]
\vspace{1cm}
\LARGE \faRocket\quad \textbf{Your Big Announcement}

\vspace{0.5cm}

\normalsize
Supporting details here
\end{frame}
```

### Use Custom Boxes
```latex
\begin{keybox}
Main takeaway or important point
\end{keybox}

\begin{ghostbox}
Ghost ads specific insight
\end{ghostbox}

\begin{warningbox}
\warningmark\ Warning or problem statement
\end{warningbox}
```

### Icon Bullets
Icons are automatically applied to itemize lists. Manual icons:
```latex
\ghostmark  → Ghost icon
\checkmark  → Green checkmark
\xmark      → Red X
\infomark   → Blue info circle
\warningmark → Orange warning triangle
```

## Conservative vs. Keynote Modes

**Current:** Keynote-ready (balanced polish)

**To make more conservative** (board meetings):
In `ghostads-theme.sty`:
1. Change line 19: `progressbar=none`
2. Remove standout frames from content files
3. Use simpler icon bullets

**To make more sexy** (launch events):
1. Enable dark mode: `\ghostadsdarkmode`
2. Add image backgrounds to title slides
3. Use geometric accents (consider switching to Trigon theme)

## Compilation Notes

- **Warnings:** Some "Overfull vbox" warnings are normal (content slightly too tall)
- **PDF size:** ~580KB for 131 pages
- **Fonts used:** Fira Sans (portable), FontAwesome 5
- **Compatibility:** Works on Overleaf, local TeXLive, MiKTeX

## Next Steps

1. **Review PDF:** Check `main.pdf` (generated on Oct 25, 2025)
2. **Adjust colors** if needed for your brand
3. **Add logo:** Uncomment logo section in `main.tex` and add logo file
4. **Optional:** Create title slide background image for extra polish
5. **Test on projector** to verify colors and contrast

## Rollback

To restore original slides:
```bash
cd ghostads
rm -rf slides
mv slides_backup_20251025_031514 slides
```

---

**Questions or customization requests?** The theme is modular and easy to adjust.

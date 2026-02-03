#!/usr/bin/env Rscript
library(ggplot2)
library(grid)

outfile <- "04_dwell_vs_load.pdf"

# Timeline events
events <- data.frame(
  time = c(0, 1.5, 5.5),
  event = c("Auction\nCreated", "Page\nLoaded", "Ad Enters\nViewport"),
  y = 1
)

p <- ggplot() +
  # Main timeline
  annotate("segment", x=-0.3, xend=6.5, y=1, yend=1, color="gray30", linewidth=1.2) +

  # Time markers (ticks)
  annotate("segment", x=0, xend=0, y=0.96, yend=1.04, color="gray30", linewidth=1) +
  annotate("segment", x=1.5, xend=1.5, y=0.96, yend=1.04, color="gray30", linewidth=1) +
  annotate("segment", x=5.5, xend=5.5, y=0.96, yend=1.04, color="gray30", linewidth=1) +

  # Event points
  annotate("point", x=c(0, 1.5, 5.5), y=1, size=5, color=c("gray50", "#0072B2", "#D55E00")) +

  # Event labels (above)
  annotate("text", x=0, y=1.15, label="Auction\nCreated", size=3.5, fontface="bold", lineheight=0.85, vjust=0) +
  annotate("text", x=1.5, y=1.15, label="Page\nLoaded", size=3.5, fontface="bold", lineheight=0.85, vjust=0, color="#0072B2") +
  annotate("text", x=5.5, y=1.15, label="Ad Enters\nViewport", size=3.5, fontface="bold", lineheight=0.85, vjust=0, color="#D55E00") +

  # Time labels (below)
  annotate("text", x=0, y=0.92, label="t = 0", size=3, color="gray40") +
  annotate("text", x=1.5, y=0.92, label="t = 1.5s", size=3, color="gray40") +
  annotate("text", x=5.5, y=0.92, label="t = 5.5s", size=3, color="gray40") +

  # === INTERVAL BARS ===
  # Load Time bar
  annotate("rect", xmin=0, xmax=1.5, ymin=0.72, ymax=0.78, fill="#0072B2", alpha=0.8) +
  annotate("text", x=0.75, y=0.75, label="Load Time", color="white", size=3.5, fontface="bold") +
  annotate("text", x=0.75, y=0.67, label="(Server + Network)", color="#0072B2", size=2.8, alpha=0.9) +

  # Dwell Time bar
  annotate("rect", xmin=1.5, xmax=5.5, ymin=0.72, ymax=0.78, fill="#D55E00", alpha=0.8) +
  annotate("text", x=3.5, y=0.75, label="Dwell Time", color="white", size=3.5, fontface="bold") +
  annotate("text", x=3.5, y=0.67, label="(User scrolls to ad position)", color="#D55E00", size=2.8, alpha=0.9) +

  # Total latency bracket
  annotate("segment", x=0, xend=0, y=0.55, yend=0.60, color="gray40", linewidth=0.5) +
  annotate("segment", x=5.5, xend=5.5, y=0.55, yend=0.60, color="gray40", linewidth=0.5) +
  annotate("segment", x=0, xend=5.5, y=0.55, yend=0.55, color="gray40", linewidth=0.5) +
  annotate("text", x=2.75, y=0.50, label="Total Latency = Load + Dwell", size=3.2, color="gray30", fontface="italic") +

  # Arrow indicating time direction
  annotate("segment", x=6.2, xend=6.5, y=1, yend=1,
           arrow=arrow(length=unit(0.15,"cm"), type="closed"), color="gray30", linewidth=1) +
  annotate("text", x=6.35, y=0.92, label="time", size=2.8, color="gray50", fontface="italic") +

  # Theme
  scale_y_continuous(limits=c(0.4, 1.4)) +
  scale_x_continuous(limits=c(-0.5, 7)) +
  theme_void() +
  theme(plot.margin = margin(10, 10, 10, 10))

ggsave(outfile, plot=p, width=7, height=3.5)
cat(paste("Saved plot to", outfile, "\n"))

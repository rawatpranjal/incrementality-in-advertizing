#!/usr/bin/env Rscript
library(ggplot2)
library(grid)

# Set output path
outfile <- "02_mobile_viewport.pdf"

# Mobile layout (2-column grid, tall phone aspect ratio):
# Row 1: Search bar
# Row 2: Ad (Rank 1) | Ad (Rank 2)
# Row 3: Organic | Organic
# --- FOLD ---
# Row 4: Ad (Rank 3) | Ad (Rank 4)

page_height <- 16
row_height <- 3.0  # Taller product cards
gap <- 0.2
col_width <- 1.8  # Narrower for phone aspect
col_gap <- 0.15
phone_width <- 4.2

# Center the content
left_xmin <- (phone_width - 2*col_width - col_gap) / 2
left_xmax <- left_xmin + col_width
right_xmin <- left_xmax + col_gap
right_xmax <- right_xmin + col_width

# Row y-coordinates (from top)
search_ymax <- page_height - 0.5
search_ymin <- search_ymax - 0.5
row1_ymax <- search_ymin - gap
row1_ymin <- row1_ymax - row_height
row2_ymax <- row1_ymin - gap
row2_ymin <- row2_ymax - row_height
row3_ymax <- row2_ymin - gap
row3_ymin <- row3_ymax - row_height

# Viewport shows search bar + row 1 (ads) + row 2 (organic)
viewport_ymin <- row2_ymin - gap/2
viewport_ymax <- page_height

slots <- data.frame(
  xmin = c(left_xmin, right_xmin, left_xmin, right_xmin, left_xmin, right_xmin),
  xmax = c(left_xmax, right_xmax, left_xmax, right_xmax, left_xmax, right_xmax),
  ymin = c(row1_ymin, row1_ymin, row2_ymin, row2_ymin, row3_ymin, row3_ymin),
  ymax = c(row1_ymax, row1_ymax, row2_ymax, row2_ymax, row3_ymax, row3_ymax),
  type = c("Sponsored", "Sponsored", "Organic", "Organic", "Sponsored", "Sponsored"),
  label = c("Rank 1", "Rank 2", "Organic", "Organic", "Rank 3", "Rank 4")
)

p <- ggplot() +
  # The full page/scroll area background
  annotate("rect", xmin=0, xmax=phone_width, ymin=0, ymax=page_height, fill="gray95", color="gray80") +

  # Search bar
  annotate("rect", xmin=left_xmin, xmax=right_xmax, ymin=search_ymin, ymax=search_ymax,
           fill="white", color="gray50", linewidth=0.3) +
  annotate("text", x=(left_xmin+right_xmax)/2, y=(search_ymin+search_ymax)/2,
           label="Search...", color="gray50", size=2.5, hjust=0.5) +

  # The slots - sponsored (gold) vs organic (white)
  geom_rect(data=slots, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fill=type),
            color="black", alpha=0.9, linewidth=0.3) +
  scale_fill_manual(values=c("Sponsored"="#FFD700", "Organic"="white")) +

  # Labels inside each slot
  geom_text(data=slots, aes(x=(xmin+xmax)/2, y=(ymin+ymax)/2, label=label),
            size=2.5, fontface="bold") +

  # The Viewport (Phone Screen) - rounded corners simulated with thick border
  annotate("rect", xmin=-0.15, xmax=phone_width+0.15, ymin=viewport_ymin, ymax=viewport_ymax+0.5,
           color="#222222", fill=NA, linewidth=2.5) +

  # Phone Notch/Dynamic Island
  annotate("rect", xmin=phone_width/2-0.6, xmax=phone_width/2+0.6, ymin=viewport_ymax+0.15, ymax=viewport_ymax+0.35,
           fill="#222222") +

  # Fold Line (between row 2 and row 3, i.e., between ad rank 2 and 3)
  annotate("segment", x=-0.8, xend=phone_width+0.8, y=viewport_ymin, yend=viewport_ymin,
           linetype="dashed", color="red", linewidth=0.8) +
  annotate("text", x=-1, y=viewport_ymin, label="Fold\n(Rank 2 vs 3)", hjust=1, color="red", fontface="bold", size=2.5) +

  # Labels for above/below fold
  annotate("text", x=phone_width+0.8, y=(viewport_ymax + viewport_ymin)/2, label="Viewport\n(Ranks 1-2)", hjust=0, fontface="bold", size=2.5) +
  annotate("text", x=phone_width+0.8, y=(row3_ymin + row3_ymax)/2, label="Below Fold\n(Ranks 3+)", hjust=0, color="gray40", fontface="bold", size=2.5) +

  # Theme
  coord_fixed(ratio=1, xlim=c(-2.5, 7), ylim=c(row3_ymin - 0.5, page_height + 1), clip="off") +
  theme_void() +
  theme(legend.position="bottom", legend.title=element_blank()) +
  labs(title="")

ggsave(outfile, plot=p, width=7, height=5)
cat(paste("Saved plot to", outfile, "\n"))

#!/usr/bin/env Rscript
library(ggplot2)
library(grid)

# Set output path
outfile <- "02_mobile_viewport.pdf"

# Data defining the "Page" layout
page_height <- 12
viewport_height <- 4  # Shows exactly 2 slots above the fold
n_slots <- 6
slot_height <- 1.5
gap <- 0.2

slots <- data.frame(
  id = 1:n_slots,
  ymin = page_height - (1:n_slots)*(slot_height + gap),
  ymax = page_height - (1:n_slots)*(slot_height + gap) + slot_height,
  type = c("Sponsored", "Organic", "Sponsored", "Organic", "Sponsored", "Organic")
)

# Viewport coordinates
viewport_ymin <- page_height - viewport_height
viewport_ymax <- page_height

p <- ggplot() +
  # The full page background
  annotate("rect", xmin=0, xmax=6, ymin=0, ymax=page_height, fill="gray95", color="gray80") +
  
  # The slots
  geom_rect(data=slots, aes(xmin=0.5, xmax=5.5, ymin=ymin, ymax=ymax, fill=type), color="black", alpha=0.9, linewidth=0.2) +
  scale_fill_manual(values=c("Sponsored"="#FFD700", "Organic"="white")) +
  
  # The Viewport (Phone Screen)
  annotate("rect", xmin=-0.2, xmax=6.2, ymin=viewport_ymin, ymax=viewport_ymax+0.5, 
           color="#333333", fill=NA, linewidth=2) + # linetype="solid" is default
  
  # Phone Notch/Bezel details
  annotate("rect", xmin=2, xmax=4, ymin=viewport_ymax+0.1, ymax=viewport_ymax+0.3, fill="#333333") +
  
  # Fold Line
  annotate("segment", x=-1, xend=7, y=viewport_ymin, yend=viewport_ymin, linetype="dashed", color="red") +
  annotate("text", x=-1.2, y=viewport_ymin, label="The Fold", hjust=1, color="red", fontface="italic", size=3) +

  # Labels
  annotate("text", x=7.2, y=(viewport_ymax + viewport_ymin)/2, label="Viewport\n(Visible Impressions)", hjust=0, fontface="bold", size=3) +
  annotate("text", x=7.2, y=viewport_ymin - 2, label='"Below the Fold"\n(Unseen until scroll)', hjust=0, color="gray50", size=3) +
  
  # Arrow pointing to viewport
  annotate("segment", x=6.2, xend=6.8, y=(viewport_ymax + viewport_ymin)/2, yend=(viewport_ymax + viewport_ymin)/2, 
           arrow=arrow(length=unit(0.2,"cm"))) +
  
  # Theme
  coord_fixed(ratio=1) +
  theme_void() +
  theme(legend.position="bottom", legend.title=element_blank()) +
  labs(title="") 

ggsave(outfile, plot=p, width=7, height=5)
cat(paste("Saved plot to", outfile, "\n"))

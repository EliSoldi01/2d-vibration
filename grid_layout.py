import matplotlib.pyplot as plt

# ----------------------------
# Parametri griglia
# ----------------------------
rows = 9
cols = 21
cell_size_cm = 4
cell_size_in = cell_size_cm / 8  # conversione cm → pollici

# Dimensione figura proporzionale alle celle
fig_width = cols * cell_size_in
fig_height = rows * cell_size_in

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# ----------------------------
# Disegno griglia
# ----------------------------
for x in range(cols + 1):
    ax.plot([x, x], [0, rows], color='black', linewidth=0.5)

for y in range(rows + 1):
    ax.plot([0, cols], [y, y], color='black', linewidth=0.5)

# ----------------------------
# Numerazione celle (riga-colonna)
# ----------------------------
highlight_row = [1, 5, 9]
highlight_col = [1, 11, 21]
for r in range(1, rows + 1):
    for c in range(1, cols + 1):
        if (r, c) in zip(highlight_row, highlight_col):
            label = f"{r}{c}"
            ax.text(c - 0.5, rows - r + 0.5, label, fontsize=12,
                    ha='center', va='center')
        else:
            continue

# ----------------------------
# Evidenzia cella 511 (riga 5, colonna 11)
# ----------------------------
highlight_row = 5
highlight_col = 11

rect = plt.Rectangle(
    (highlight_col - 1, rows - highlight_row),
    1, 1,
    fill=False,
    linewidth=3,
    color='red',
    label="Starting position"
)

ax.add_patch(rect)

# ----------------------------
# Indicazione dimensione cella (4 cm)
# ----------------------------
ax.annotate(
    "",
    xy=(1, rows - 0.5),
    xytext=(3, rows - 0.5),
    arrowprops=dict(arrowstyle="<|-")
)
ax.text(1.7, rows-0.4, "x", fontsize=14, ha='center')
ax.annotate(
    "",
    xy=(0.5, rows - 1),
    xytext=(0.5, rows - 3),
    arrowprops=dict(arrowstyle="<|-")
)
ax.text(0.3, rows-1.7, "y", fontsize=14, ha='center')

# Larghezza
ax.annotate(
    "",
    xy=(0, -0.3),
    xytext=(1, -0.3),
    arrowprops=dict(arrowstyle="<->")
)
ax.text(0.5, -0.9, "4 cm", fontsize=14, ha='center')

# Altezza
ax.annotate(
    "",
    xy=(-0.3, 0),
    xytext=(-0.3, 1),
    arrowprops=dict(arrowstyle="<->")
)
ax.text(-0.9, 0.5, "4 cm", fontsize=14, rotation=90, va='center')

# ----------------------------
# Impostazioni finali
# ----------------------------
margin_x = 1
margin_y = 1

ax.set_xlim(-margin_x, cols + margin_x)
ax.set_ylim(-margin_y, rows + margin_y)

ax.set_aspect('equal')
ax.axis('off')

fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    fontsize=14,
    frameon=False
)
plt.tight_layout()
plt.savefig("grid_9x21.png", dpi=300, bbox_inches="tight")
plt.show()
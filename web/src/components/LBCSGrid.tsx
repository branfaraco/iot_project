const LBCSGrid = () => {
  // Sample LBCS grid data. In a full integration this would be the one‑hot
  // encoded land‑use categories for each cell (e.g. residential, commercial).
  const gridData = Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, () => Math.random())
  );

  const getColor = (value: number) => {
    const hue = 172 + value * 27; // Accent range
    return `hsl(${hue}, 66%, ${30 + value * 30}%)`;
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        This grid depicts the land‑use categories (LBCS) used to enrich the
        prediction model. Each cell’s colour encodes the relative importance of
        that land‑use class.
      </p>
      <div className="bg-muted/50 p-6 rounded-lg border border-border">
        <div className="grid grid-cols-8 gap-1 max-w-md mx-auto">
          {gridData.map((row, rowIndex) =>
            row.map((cell, colIndex) => (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="aspect-square rounded-sm flex items-center justify-center text-[9px] font-mono transition-all hover:scale-110"
                style={{ backgroundColor: getColor(cell) }}
              >
                {cell.toFixed(2)}
              </div>
            ))
          )}
        </div>
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
        <span>Grid Resolution: 8×8</span>
        <span>Note: sample values shown</span>
      </div>
    </div>
  );
};

export default LBCSGrid;
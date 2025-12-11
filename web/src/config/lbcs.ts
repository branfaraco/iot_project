// Official LBCS 1‑digit function colour codes and descriptions.
//
// Each key corresponds to a 1‑digit LBCS function code.  The values are
// the recommended hexadecimal colours and human‑readable descriptions.  A
// helper is also provided to convert hex strings into RGBA arrays for
// deck.gl layers.

export const LBCS_COLOR_HEX: Record<string, string> = {
  "1000": "#FFFF00", // yellow
  "2000": "#FF0000", // red
  // Note: the official RGB values for code 3000 are (160, 32, 240).  The
  // hex code in some documents appears as "A0F020" but that is a scan
  // error (swapping the green and blue channels).  The correct hex for
  // (160, 32, 240) is A020F0, which is a bright purple.  See colour
  // reference: https://www.color-hex.com/color/a020f0
  "3000": "#A020F0", // purple
  "4000": "#BEBEBE", // gray
  "5000": "#90EE90", // light green
  "6000": "#0000FF", // blue
  "7000": "#008B8B", // dark cyan
  "8000": "#551A8B", // purple4
  "9000": "#228B22", // forest green
  "0000": "#FFFFFF", // unclassified / unknown
};

export const LBCS_DESCRIPTION: Record<string, string> = {
  "1000": "Residence or accommodation functions",
  "2000": "General sales or services",
  "3000": "Manufacturing and wholesale trade",
  "4000": "Transportation, communication, information, and utilities",
  "5000": "Arts, entertainment, and recreation",
  "6000": "Education, public administration, health care, and other institutions",
  "7000": "Construction-related businesses",
  "8000": "Mining and extraction establishments",
  "9000": "Agriculture, forestry, fishing, and hunting",
  "0000": "Unclassified / unknown",
};

/**
 * Convert a hex colour string to an RGBA array.  deck.gl expects fill
 * colours as [r, g, b, a] where a is an integer 0–255.  Use this helper
 * when constructing layers.
 */
export function hexToRGBA(hex: string, alpha = 180): [number, number, number, number] {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return [r, g, b, alpha];
}
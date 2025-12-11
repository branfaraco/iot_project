// src/pages/LbcsMap.tsx
//
// Page for rendering the static LBCS grid as a map.  This page
// loads the GeoJSON from `public/lbcs.geojson`, computes a view
// state centred on the average of all coordinates, and renders the
// polygons using a GeoJsonLayer.  Colours and descriptions are
// defined in `src/config/lbcs.ts`.

import React, { useEffect, useState } from "react";
import { GeoJsonLayer } from "@deck.gl/layers";
import MapBase, { ViewState } from "@/components/MapBase";
import type { FeatureCollection } from "geojson";
import { LBCS_COLOR_HEX, LBCS_DESCRIPTION, hexToRGBA } from "@/config/lbcs";

const LbcsMap: React.FC = () => {
  const [data, setData] = useState<FeatureCollection | null>(null);
  const [viewState, setViewState] = useState<ViewState>({
    latitude: 0,
    longitude: 0,
    zoom: 10,
    pitch: 0,
    bearing: 0,
    orthographic: false,
  });

  useEffect(() => {
    // Load GeoJSON from the public folder.  Vite serves files under
    // `public/` at the root URL during development and as static assets
    // in production.
    fetch("/lbcs.geojson")
      .then((res) => res.json())
      .then((fc: FeatureCollection) => {
        setData(fc);
        // Compute a simple centroid by averaging all coordinates so
        // the map view is centred over the dataset.
        const coords: [number, number][] = [];
        for (const feature of fc.features) {
          const geom: any = feature.geometry;
          if (!geom) continue;
          const pushCoords = (c: any) => {
            if (Array.isArray(c[0])) {
              c.forEach(pushCoords);
            } else if (typeof c[0] === "number" && typeof c[1] === "number") {
              coords.push([c[1], c[0]]);
            }
          };
          pushCoords(geom.coordinates);
        }
        if (coords.length > 0) {
          const avgLat = coords.reduce((sum, c) => sum + c[0], 0) / coords.length;
          const avgLon = coords.reduce((sum, c) => sum + c[1], 0) / coords.length;
          setViewState((vs) => ({ ...vs, latitude: avgLat, longitude: avgLon }));
        }
      })
      .catch((err) => {
        console.error("Failed to load /lbcs.geojson", err);
      });
  }, []);

  const onViewStateChange = ({ viewState }: { viewState: ViewState }) => {
    setViewState(viewState);
  };

  const renderLayers = () => {
    if (!data) return [];
    const layer = new GeoJsonLayer({
      id: "lbcs-layer",
      data,
      pickable: true,
      stroked: true,
      filled: true,
      extruded: false,
      getFillColor: (feature: any) => {
        const code: string | undefined = feature.properties?.lbcs_first_level;
        const hex = (code && LBCS_COLOR_HEX[code]) || LBCS_COLOR_HEX["0000"];
        return hexToRGBA(hex, 200);
      },
      getLineColor: [0, 0, 0, 80],
      lineWidthMinPixels: 1,
    });
    return [layer];
  };

  return (
    <div className="w-screen h-screen relative">
      <MapBase
        viewState={viewState}
        onViewStateChange={onViewStateChange}
        renderLayers={renderLayers}
      />
      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/90 rounded-md shadow p-3 text-xs space-y-1 text-black  max-w-xs">
        <div className="font-semibold mb-1">LBCS Legend</div>
        {Object.keys(LBCS_COLOR_HEX).map((code) => (
          <div key={code} className="flex items-start gap-2 mb-1">
            <span
              className="inline-block w-3 h-3 mt-1 rounded-sm border"
              style={{ backgroundColor: LBCS_COLOR_HEX[code] }}
            />
            <div className="flex flex-col">
              <span className="font-mono">{code}</span>
              <span className="text-[10px]">{LBCS_DESCRIPTION[code]}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LbcsMap;
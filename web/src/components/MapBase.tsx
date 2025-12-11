// src/components/MapBase.tsx
//
// Reusable map component built on top of deck.gl and react-map-gl.
//
// This component wraps DeckGL and a Mapbox map.  It accepts a view
// state, a callback to update the view state, and a function that
// returns an array of deck.gl layers.  A click handler may also be
// provided.  See pages/LbcsMap.tsx for a usage example.

import React, { useEffect, useState, useRef } from "react";
import DeckGL from "@deck.gl/react";
import { StaticMap } from "react-map-gl";
import "mapbox-gl/dist/mapbox-gl.css";

export interface ViewState {
  latitude: number;
  longitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
  orthographic: boolean;
}

interface MapBaseProps {
  viewState: ViewState;
  onViewStateChange: ({ viewState }: { viewState: ViewState }) => void;
  /**
   * Function that returns deck.gl layers.  The argument is a no-op
   * hover callback which you may ignore or replace inside your
   * implementation.  See pages/LbcsMap.tsx for an example.
   */
  renderLayers: (handleHover: (info: any) => void) => any[];
  onClick?: (info: any) => void;
}

const MapBase: React.FC<MapBaseProps> = ({ viewState, onViewStateChange, renderLayers, onClick }) => {
  const deckGLref = useRef<any>(null);
  const [draggingWhileEditing, setDraggingWhileEditing] = useState<boolean>(false);

  useEffect(() => {
    // Prevent context menu on right click inside the map.
    const deckWrapper = document.getElementById("deckgl-wrapper");
    if (deckWrapper) {
      const handler = (evt: MouseEvent) => evt.preventDefault();
      deckWrapper.addEventListener("contextmenu", handler);
      return () => {
        deckWrapper.removeEventListener("contextmenu", handler);
      };
    }
  }, []);

  const layers = renderLayers(() => {});

  return (
    <div id="deckgl-wrapper">
      <DeckGL
        ref={deckGLref}
        viewState={viewState}
        onViewStateChange={onViewStateChange}
        layers={layers}
        controller={{
          dragPan: !draggingWhileEditing,
          dragRotate: !draggingWhileEditing,
          keyboard: false,
        }}
        onClick={onClick}
      >
        <StaticMap
  reuseMaps
  mapStyle="mapbox://styles/relnox/ck0h5xn701bpr1dqs3he2lecq?fresh=true"
  mapboxApiAccessToken={(import.meta as any).env.VITE_MAPBOX_TOKEN}
/>
      </DeckGL>
    </div>
  );
};

export default MapBase;
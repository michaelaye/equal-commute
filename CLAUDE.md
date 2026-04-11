# Equal Commute Time Finder — Berlin

## Intent

Find areas in Berlin where driving commute time to two workplaces is approximately equal (within ±5 min), to help choose where to live.

### Target locations

- **A**: Rutherfordstraße 2, 12489 Berlin (Adlershof)
- **B**: Preysingstraße 53, 12249 Berlin (Lankwitz)

## Approach

1. Sample a grid of points across Berlin
2. Query OpenRouteService (ORS) Matrix API for driving times from each grid point to both destinations
3. Compute the scalar field of time differences (t_A − t_B)
4. Extract the zero contour (equal commute boundary) and a ±5 min tolerance band
5. Visualize on an interactive Folium/Leaflet map with red↔blue heatmap overlay

## Key files

- `equi_commute.py` — main script, outputs `equi_commute_berlin.html`

## Configuration

- ORS API key in env var `ORS_API_KEY` (free tier, set via `set -Ux` in fish)
- `GRID_RESOLUTION` controls accuracy vs. API usage (40 = quick, 80 = smooth)
- `MAX_DIFF_MINUTES` controls the tolerance band width (default 5)
- Coordinates for A/B are approximate — could be improved with ORS geocoding

## Possible next steps

- Add ORS geocoding to resolve exact addresses automatically
- Support public transit mode (ORS doesn't; would need a different backend)
- Overlay POIs (schools, Kitas, parks, supermarkets)
- Export the equal-time contour as GeoJSON for use in QGIS or similar
- Add isochrone polygons from both locations as additional map layers
- Filter the tolerance band to only include residential areas

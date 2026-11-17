"""
Compute the equal-commute-time boundary between two locations in Berlin
using the OpenRouteService Matrix API.

Usage:
    export ORS_API_KEY="your-key-here"
    python equi_commute.py

Outputs an interactive HTML map.
"""

import os
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import requests
from scipy.interpolate import griddata
from shapely.geometry import shape, MultiPolygon, Polygon
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib import contour as mcontour
import folium
from folium.plugins import HeatMap, MarkerCluster, Geocoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORS_API_KEY = os.environ.get("ORS_API_KEY", "")
if not ORS_API_KEY:
    raise SystemExit("Set ORS_API_KEY environment variable first.")

ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/{profile}"
ORS_ISOCHRONE_URL = "https://api.openrouteservice.org/v2/isochrones/driving-car"

# Modes shown in POI popups — (profile, display label)
POI_TRAVEL_MODES = [
    ("driving-car",    "drive"),
    ("cycling-regular", "bike"),
    ("foot-walking",   "walk"),
]

# Target locations [lon, lat]  (ORS uses lon,lat order)
LOCATION_A = (13.5285, 52.4310)   # Rutherfordstraße 2, 12489 Berlin (Adlershof)
LOCATION_B = (13.3575, 52.4275)   # Preysingstraße 53, 12249 Berlin (Lankwitz)

# Grid definition over Berlin
BERLIN_BBOX = {
    "lon_min": 13.08,
    "lon_max": 13.77,
    "lat_min": 52.22,
    "lat_max": 52.68,
}

GRID_RESOLUTION = 120  # points per axis → 14400 grid points
MAX_DIFF_MINUTES = 5  # tolerance band: |t_A - t_B| < this value
MAX_SNAP_DISTANCE_M = 400  # drop grid points ORS snapped > this far from the road

# Addresses of interest — pinned markers geocoded via Nominatim (cached).
ADDRESSES_OF_INTEREST = [
    "Kienitzer Str. 28, 15834 Rangsdorf, Germany",
]

# POI proximity: for each address of interest, pre-compute driving time to
# every POI within this straight-line radius. Results shown in marker popup.
POI_PROXIMITY_RADIUS_KM = 15

# ORS matrix API limit: 3500 sources × 1 destination per request,
# but free tier caps at 2500 per request. We batch at 2000.
BATCH_SIZE = 2000
REQUEST_DELAY = 1.5  # seconds between API calls (free tier: 40 req/min)


# ---------------------------------------------------------------------------
# Build sample grid
# ---------------------------------------------------------------------------
def build_grid():
    """Create a regular lon/lat grid over Berlin."""
    lons = np.linspace(BERLIN_BBOX["lon_min"], BERLIN_BBOX["lon_max"], GRID_RESOLUTION)
    lats = np.linspace(BERLIN_BBOX["lat_min"], BERLIN_BBOX["lat_max"], GRID_RESOLUTION)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    return lons, lats, points


# ---------------------------------------------------------------------------
# ORS Matrix queries
# ---------------------------------------------------------------------------
def query_matrix_batched(grid_points, destination, profile="driving-car"):
    """
    Query ORS matrix API for travel times from all grid points to a single
    destination.  Returns (durations, snap_distances) — both length-n arrays,
    NaN where the API failed.  snap_distances is the distance in meters from
    the requested point to the road ORS snapped it to (large = in water,
    forest, or otherwise inaccessible).
    """
    n = len(grid_points)
    durations = np.full(n, np.nan)
    snap_distances = np.full(n, np.nan)

    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_points = grid_points[batch_start:batch_end]

        # Build locations: batch points + destination (last index)
        locations = batch_points.tolist() + [list(destination)]
        sources = list(range(len(batch_points)))
        destinations = [len(batch_points)]  # destination is last

        payload = {
            "locations": locations,
            "sources": sources,
            "destinations": destinations,
            "metrics": ["duration"],
            "resolve_locations": True,  # makes ORS return snapped_distance per source
        }
        headers = {
            "Authorization": ORS_API_KEY,
            "Content-Type": "application/json; charset=utf-8",
        }

        resp = requests.post(ORS_MATRIX_URL.format(profile=profile), json=payload, headers=headers)
        if resp.status_code != 200:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            time.sleep(REQUEST_DELAY)
            continue

        data = resp.json()
        batch_durations = [row[0] for row in data["durations"]]
        sources_meta = data.get("sources", [])
        for i, d in enumerate(batch_durations):
            if d is not None:
                durations[batch_start + i] = d
            if i < len(sources_meta) and sources_meta[i] is not None:
                snap = sources_meta[i].get("snapped_distance")
                if snap is not None:
                    snap_distances[batch_start + i] = snap

        done = batch_end
        print(f"  {done}/{n} points queried")
        time.sleep(REQUEST_DELAY)

    return durations, snap_distances


OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def _overpass_post(body, timeout=240):
    """POST to Overpass with endpoint fallback. Raises on total failure."""
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(url, data={"data": body}, timeout=timeout)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
            last_err = f"{url}: HTTP {resp.status_code} / ct={resp.headers.get('content-type')} / body={resp.text[:120]}"
            print(f"  Overpass endpoint failed → {last_err}")
        except Exception as e:
            last_err = f"{url}: {e}"
            print(f"  Overpass endpoint error → {last_err}")
    raise RuntimeError(f"All Overpass endpoints failed. Last: {last_err}")

POI_CATEGORIES = {
    # category slug → (osm_key, osm_value, display name, folium color, fa icon)
    "kindergarten": ("amenity", "kindergarten", "Kita",       "green",     "child"),
    "school":       ("amenity", "school",       "School",     "darkgreen", "graduation-cap"),
    "restaurant":   ("amenity", "restaurant",   "Restaurant", "orange",    "utensils"),
    "cafe":         ("amenity", "cafe",         "Cafe",       "beige",     "coffee"),
    "supermarket":  ("shop",    "supermarket",  "Groceries",  "red",       "shopping-cart"),
}


def query_overpass_pois(bbox):
    """Fetch POIs from OpenStreetMap for every entry in POI_CATEGORIES.

    Returns dict: {category_slug: [(lat, lon, name), ...]}.
    """
    s, w, n, e = bbox["lat_min"], bbox["lon_min"], bbox["lat_max"], bbox["lon_max"]
    clauses = []
    for slug, (osm_key, osm_value, *_rest) in POI_CATEGORIES.items():
        clauses.append(f'node["{osm_key}"="{osm_value}"]({s},{w},{n},{e});')
        clauses.append(f'way["{osm_key}"="{osm_value}"]({s},{w},{n},{e});')
    body = (
        "[out:json][timeout:120];\n"
        "(\n" + "\n".join(clauses) + "\n);\n"
        "out center tags;"
    )
    payload = _overpass_post(body, timeout=180)

    # Map (osm_key, osm_value) → slug for classification
    lookup = {(k, v): slug for slug, (k, v, *_r) in POI_CATEGORIES.items()}
    result = {slug: [] for slug in POI_CATEGORIES}
    for el in payload.get("elements", []):
        tags = el.get("tags", {})
        slug = None
        for (k, v), s in lookup.items():
            if tags.get(k) == v:
                slug = s
                break
        if slug is None:
            continue
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            c = el.get("center")
            if not c:
                continue
            lat, lon = c["lat"], c["lon"]
        name = tags.get("name", "")
        result[slug].append((lat, lon, name))
    return result


def haversine_km_vec(lat1, lon1, lats2, lons2):
    """Vectorized haversine distance in km from one point to many."""
    R = 6371.0
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lats2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lons2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_poi_proximity(pois, addresses, radius_km=POI_PROXIMITY_RADIUS_KM):
    """For each address, compute driving/cycling/walking time + distance per POI.

    Returns nested dict
      {(lat, lon, name, slug): {address_str: {"dist_km": float, "drive": sec, "bike": sec, "walk": sec}}}
    Each mode value is None if routing failed.
    """
    all_pois = []
    for slug, items in pois.items():
        for lat, lon, name in items:
            all_pois.append((lat, lon, name, slug))
    if not all_pois or not addresses:
        return {}

    all_lats = np.array([p[0] for p in all_pois])
    all_lons = np.array([p[1] for p in all_pois])

    result = {(p[0], p[1], p[2], p[3]): {} for p in all_pois}

    for addr, a_lat, a_lon, _display in addresses:
        dists_km = haversine_km_vec(a_lat, a_lon, all_lats, all_lons)
        nearby = np.where(dists_km <= radius_km)[0]
        if len(nearby) == 0:
            print(f"  No POIs within {radius_km} km of '{addr}'")
            continue

        nearby_coords = np.column_stack([all_lons[nearby], all_lats[nearby]])
        per_mode = {}
        for profile, label in POI_TRAVEL_MODES:
            print(f"  Routing {len(nearby)} POIs to '{addr[:40]}' via {label}...")
            durations, _ = query_matrix_batched(nearby_coords, (a_lon, a_lat), profile=profile)
            per_mode[label] = durations

        for local_i, global_i in enumerate(nearby):
            p = all_pois[global_i]
            entry = {"dist_km": float(dists_km[global_i])}
            for _prof, label in POI_TRAVEL_MODES:
                d = per_mode[label][local_i]
                entry[label] = float(d) if not np.isnan(d) else None
            result[(p[0], p[1], p[2], p[3])][addr] = entry
    return result


def load_or_compute_poi_proximity(pois, addresses, cache_path):
    """Cache proximity results by a fingerprint of POI coordinates + addresses."""
    coords = sorted((round(lat, 6), round(lon, 6))
                    for items in pois.values() for lat, lon, _ in items)
    fp = hashlib.sha1(json.dumps({
        "pois": hashlib.sha1(str(coords).encode()).hexdigest()[:12],
        "addresses": [a[0] for a in addresses],
        "radius_km": POI_PROXIMITY_RADIUS_KM,
        "modes": [label for _p, label in POI_TRAVEL_MODES],
    }, sort_keys=True).encode()).hexdigest()[:12]

    if cache_path.exists():
        with open(cache_path) as f:
            blob = json.load(f)
        if blob.get("fingerprint") == fp:
            print(f"Loaded cached POI proximities from {cache_path.name}")
            out = {}
            for key_str, addrs in blob["proximity"].items():
                key = tuple(json.loads(key_str))
                out[key] = {a: dict(entry) for a, entry in addrs.items()}
            return out

    print("Computing POI proximities...")
    prox = compute_poi_proximity(pois, addresses)

    serializable = {
        json.dumps(list(k)): {a: entry for a, entry in addrs.items()}
        for k, addrs in prox.items() if addrs
    }
    with open(cache_path, "w") as f:
        json.dump({"fingerprint": fp, "proximity": serializable}, f)
    print(f"Cached → {cache_path.name}")
    return prox


def geocode_addresses(addresses, cache_path):
    """Geocode a list of addresses via Nominatim, caching results by address string."""
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    results = []
    for addr in addresses:
        if addr in cache:
            lat, lon, display = cache[addr]
            results.append((addr, lat, lon, display))
            continue
        print(f"  Geocoding '{addr}'...")
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": addr, "format": "json", "limit": 1},
            headers={"User-Agent": "equi-commute-script/1.0"},
            timeout=30,
        )
        if resp.status_code != 200 or not resp.json():
            print(f"  WARNING: geocoding failed for '{addr}'")
            continue
        item = resp.json()[0]
        lat, lon = float(item["lat"]), float(item["lon"])
        display = item.get("display_name", addr)
        cache[addr] = [lat, lon, display]
        results.append((addr, lat, lon, display))
        time.sleep(1.0)  # Nominatim rate limit: 1 req/s

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    return results


def query_osm_water(bbox):
    """Fetch water polygons (natural=water) from OSM via Overpass."""
    s, w, n, e = bbox["lat_min"], bbox["lon_min"], bbox["lat_max"], bbox["lon_max"]
    body = (
        "[out:json][timeout:180];\n"
        "(\n"
        f'  way["natural"="water"]({s},{w},{n},{e});\n'
        f'  relation["natural"="water"]({s},{w},{n},{e});\n'
        ");\n"
        "out geom;"
    )
    payload = _overpass_post(body, timeout=240)

    polys = []
    for el in payload.get("elements", []):
        if el["type"] == "way":
            coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                polys.append(Polygon(coords))
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") != "outer":
                    continue
                coords = [(p["lon"], p["lat"]) for p in m.get("geometry", [])]
                if len(coords) < 4:
                    continue
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                try:
                    polys.append(Polygon(coords))
                except Exception:
                    pass
    return polys


def load_or_query_water(cache_path):
    """Load cached water polygons if bbox matches, else query and save."""
    bbox_fp = hashlib.sha1(
        json.dumps(BERLIN_BBOX, sort_keys=True).encode()
    ).hexdigest()[:12]

    if cache_path.exists():
        with open(cache_path) as f:
            blob = json.load(f)
        if blob.get("fingerprint") == bbox_fp:
            print(f"Loaded cached water polygons from {cache_path.name}")
            return [Polygon(ring) for ring in blob["polys"] if len(ring) >= 4]

    print("Querying OpenStreetMap for water polygons...")
    polys = query_osm_water(BERLIN_BBOX)
    print(f"  Found {len(polys)} water polygons")
    if not polys:
        print("  WARNING: empty result, not caching")
        return polys
    with open(cache_path, "w") as f:
        json.dump({
            "fingerprint": bbox_fp,
            "polys": [list(map(list, p.exterior.coords)) for p in polys],
        }, f)
    return polys


def build_water_mask(grid_points, water_polys):
    """Return boolean array — True where a grid point lies inside any water polygon."""
    if not water_polys:
        return np.zeros(len(grid_points), dtype=bool)
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    tree = STRtree(water_polys)
    mask = np.zeros(len(grid_points), dtype=bool)
    for i, (lon, lat) in enumerate(grid_points):
        pt = Point(lon, lat)
        for idx in tree.query(pt):
            if water_polys[idx].contains(pt):
                mask[i] = True
                break
    return mask


def load_or_query_pois(cache_path):
    """Load cached OSM POIs if bbox+categories match, else fetch and save."""
    # Include the category list so adding/removing a category auto-invalidates
    bbox_fp = hashlib.sha1(
        json.dumps(
            {"bbox": BERLIN_BBOX, "cats": sorted(POI_CATEGORIES.keys())},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]

    if cache_path.exists():
        with open(cache_path) as f:
            blob = json.load(f)
        if blob.get("fingerprint") == bbox_fp:
            print(f"Loaded cached POIs from {cache_path.name} (fp={bbox_fp})")
            return {k: [tuple(x) for x in v] for k, v in blob["pois"].items()}
        print(f"POI cache fingerprint mismatch; re-querying Overpass")

    print("Querying OpenStreetMap (Overpass) for POIs...")
    pois = query_overpass_pois(BERLIN_BBOX)
    for cat, items in pois.items():
        print(f"  {cat}: {len(items)}")

    total = sum(len(v) for v in pois.values())
    if total == 0:
        print("  WARNING: empty result, not caching")
        return pois

    with open(cache_path, "w") as f:
        json.dump({"fingerprint": bbox_fp, "pois": pois}, f)
    print(f"Cached POIs → {cache_path.name}")
    return pois


def query_isochrones(location, intervals_minutes=(10, 20, 30, 45, 60)):
    """Fetch isochrone polygons for sanity-check visualization."""
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "locations": [list(location)],
        "range": [m * 60 for m in intervals_minutes],
        "range_type": "time",
    }
    resp = requests.post(ORS_ISOCHRONE_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        print(f"Isochrone error {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()


# ---------------------------------------------------------------------------
# Compute equal-time contour
# ---------------------------------------------------------------------------
def compute_contour(lons, lats, durations_a, durations_b):
    """
    Interpolate travel-time difference and average commute time onto a
    regular grid. Returns (lon_fine, lat_fine, lon_mesh, lat_mesh,
    diff_grid, time_grid) — diff in signed minutes (A-B), time in minutes.

    Uses LINEAR interpolation (no cubic overshoot) and marks fine-grid
    cells that are too far from any valid source point as NaN, so large
    forest/water holes appear as empty regions rather than interpolated
    fantasies.
    """
    from scipy.spatial import cKDTree

    diff = (durations_a - durations_b) / 60.0               # minutes
    avg_time = (durations_a + durations_b) / 2.0 / 60.0     # minutes

    lon_fine = np.linspace(BERLIN_BBOX["lon_min"], BERLIN_BBOX["lon_max"], 200)
    lat_fine = np.linspace(BERLIN_BBOX["lat_min"], BERLIN_BBOX["lat_max"], 200)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)

    valid = ~np.isnan(diff) & ~np.isnan(avg_time)
    points_valid = np.column_stack([
        np.tile(lons, len(lats))[valid],
        np.repeat(lats, len(lons))[valid],
    ])

    diff_grid = griddata(points_valid, diff[valid], (lon_mesh, lat_mesh), method="linear")
    time_grid = griddata(points_valid, avg_time[valid], (lon_mesh, lat_mesh), method="linear")

    # Mark fine-grid cells that don't have enough nearby valid sources.
    # A lone valid point in a forest interior can interpolate outward and
    # create ghost regions, so we require the 4th-nearest neighbor to be
    # within ~1.5× raw grid spacing. Cells that fail this (isolated points
    # or sparse regions) become NaN.
    raw_step = 0.5 * (
        (BERLIN_BBOX["lon_max"] - BERLIN_BBOX["lon_min"]) / GRID_RESOLUTION
        + (BERLIN_BBOX["lat_max"] - BERLIN_BBOX["lat_min"]) / GRID_RESOLUTION
    )
    max_dist = 2.0 * raw_step

    tree = cKDTree(points_valid)
    query_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    distances, _ = tree.query(query_points, k=1)
    far_mask = (distances > max_dist).reshape(lon_mesh.shape)

    diff_grid[far_mask] = np.nan
    time_grid[far_mask] = np.nan

    n_far = int(far_mask.sum())
    print(f"  {n_far} fine-grid cells marked as no-data (>{max_dist:.4f}° from any valid source)")

    return lon_fine, lat_fine, lon_mesh, lat_mesh, diff_grid, time_grid


def extract_contour_coords(lon_fine, lat_fine, diff_grid, level=0.0):
    """Extract contour line coordinates at given level."""
    fig, ax = plt.subplots()
    cs = ax.contour(lon_fine, lat_fine, diff_grid, levels=[level])
    plt.close(fig)

    lines = []
    for seg in cs.allsegs[0]:
        lines.append(seg.tolist())  # list of [lon, lat] pairs
    return lines


def filled_contour_polygons(x, y, z, lower, upper, mask=None):
    """Extract filled contour polygons between two levels.

    Returns a list of polygons, where each polygon is a list of rings
    (first ring = outer boundary, subsequent rings = holes), each ring a
    list of [lat, lon] pairs ready for folium.Polygon.

    If `mask` is provided (True = invalid cell), it is applied via a numpy
    masked array so contourpy skips any quad touching an invalid corner.
    """
    from contourpy import contour_generator, FillType
    if mask is not None:
        z = np.ma.array(z, mask=mask)
    gen = contour_generator(x=x, y=y, z=z, fill_type=FillType.OuterCode)
    points_list, codes_list = gen.filled(lower, upper)

    polygons = []
    for pts, codes in zip(points_list, codes_list):
        rings, current = [], []
        for pt, code in zip(pts, codes):
            if code == 1 and current:  # MOVETO → new ring
                rings.append(current)
                current = []
            current.append([float(pt[1]), float(pt[0])])  # lat, lon
        if current:
            rings.append(current)
        if rings:
            polygons.append(rings)
    return polygons


# ---------------------------------------------------------------------------
# Map visualization
# ---------------------------------------------------------------------------
def build_map(
    diff_grid, time_grid, lon_fine, lat_fine,
    contour_lines, isochrones_a, isochrones_b,
    pois=None, addresses=None, poi_proximity=None,
):
    """Build an interactive Folium map."""
    center_lat = (BERLIN_BBOX["lat_min"] + BERLIN_BBOX["lat_max"]) / 2
    center_lon = (BERLIN_BBOX["lon_min"] + BERLIN_BBOX["lon_max"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    # ---- Travel-time difference as filled contour bands ----
    # In-band (±MAX_DIFF_MINUTES) is left transparent so the basemap shows
    # through and the decision zone pops. Out-of-band is shaded red↔blue.
    BIG = 1e6
    bands = [
        (-BIG, -30.0,              "#67001f", 0.55),  # very fast to A
        (-30.0, -15.0,             "#d6604d", 0.50),
        (-15.0, -MAX_DIFF_MINUTES, "#f4a582", 0.40),
        # in-band gap (transparent)
        (MAX_DIFF_MINUTES,  15.0,  "#92c5de", 0.40),
        ( 15.0, 30.0,              "#4393c3", 0.50),
        ( 30.0,  BIG,              "#2166ac", 0.55),
    ]
    # contourpy needs a finite, NaN-free grid; fill NaNs with a sentinel
    # outside any band so they become empty.
    z = np.where(np.isnan(diff_grid), BIG * 2, diff_grid)
    for lower, upper, color, opacity in bands:
        for rings in filled_contour_polygons(lon_fine, lat_fine, z, lower, upper):
            folium.Polygon(
                locations=rings,
                color=color, weight=0,
                fill=True, fill_color=color, fill_opacity=opacity,
            ).add_to(m)

    # ---- Commute-time colormap inside the ±MAX_DIFF_MINUTES band ----
    # Mask cells outside the band so contourpy only draws inside.
    invalid = (
        np.isnan(diff_grid)
        | np.isnan(time_grid)
        | (np.abs(diff_grid) > MAX_DIFF_MINUTES)
    )
    time_z = np.where(np.isnan(time_grid), 0.0, time_grid)
    time_levels = [0, 15, 20, 25, 30, 35, 40, 50, 120]
    time_cmap = plt.cm.viridis
    time_norm = plt.Normalize(vmin=15, vmax=45)
    for lo, hi in zip(time_levels[:-1], time_levels[1:]):
        mid = (lo + hi) / 2
        color = matplotlib.colors.to_hex(time_cmap(time_norm(mid)))
        for rings in filled_contour_polygons(
            lon_fine, lat_fine, time_z, lo, hi, mask=invalid,
        ):
            folium.Polygon(
                locations=rings,
                color=color, weight=0,
                fill=True, fill_color=color, fill_opacity=0.18,
                tooltip=f"{lo}-{hi} min commute",
            ).add_to(m)

    # ---- Equal-time contour ----
    for line in contour_lines:
        coords_latlon = [[p[1], p[0]] for p in line]  # folium uses lat,lon
        folium.PolyLine(
            coords_latlon,
            color="black",
            weight=3,
            dash_array="5",
            tooltip="Equal commute time",
        ).add_to(m)

    # ---- Tolerance band contours (±MAX_DIFF_MINUTES) ----
    for level, style in [
        (-MAX_DIFF_MINUTES, {"color": "gray", "dash_array": "2 6"}),
        (MAX_DIFF_MINUTES, {"color": "gray", "dash_array": "2 6"}),
    ]:
        band_lines = extract_contour_coords(
            np.linspace(BERLIN_BBOX["lon_min"], BERLIN_BBOX["lon_max"], 200),
            np.linspace(BERLIN_BBOX["lat_min"], BERLIN_BBOX["lat_max"], 200),
            diff_grid,
            level=level,
        )
        for line in band_lines:
            coords_latlon = [[p[1], p[0]] for p in line]
            folium.PolyLine(
                coords_latlon, weight=1.5,
                tooltip=f"±{MAX_DIFF_MINUTES} min band",
                **style,
            ).add_to(m)

    # ---- POI overlays (toggleable via layer control) ----
    if pois:
        for cat, items in pois.items():
            if not items:
                continue
            _key, _val, display, color, icon = POI_CATEGORIES[cat]
            fg = folium.FeatureGroup(name=f"{display} ({len(items)})", show=False)
            cluster = MarkerCluster().add_to(fg)
            for lat, lon, name in items:
                popup_html = f"<b>{name or display}</b>"
                prox = (poi_proximity or {}).get((lat, lon, name, cat), {})
                if prox:
                    sections = ""
                    for addr, entry in prox.items():
                        short = addr.split(",")[0]
                        dist_km = entry.get("dist_km")
                        parts = [f"{dist_km:.1f} km"] if dist_km is not None else []
                        for _p, label in POI_TRAVEL_MODES:
                            sec = entry.get(label)
                            if sec is not None:
                                parts.append(f"{label} {sec/60:.0f} min")
                        sections += (
                            f"<div style='margin-top:4px;'>"
                            f"<b>{short}</b><br>{' &middot; '.join(parts)}</div>"
                        )
                    popup_html += f"<div style='font-size:11px;'>{sections}</div>"
                folium.Marker(
                    [lat, lon],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=color, icon=icon, prefix="fa"),
                ).add_to(cluster)
            fg.add_to(m)

    # ---- Pinned addresses of interest ----
    if addresses:
        for addr, lat, lon, display in addresses:
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(f"<b>{addr}</b><br>{display}", max_width=300),
                tooltip=addr,
                icon=folium.Icon(color="purple", icon="star", prefix="fa"),
            ).add_to(m)

    # ---- Markers for A and B ----
    folium.Marker(
        [LOCATION_A[1], LOCATION_A[0]],
        popup="A: Rutherfordstr. 2 (Adlershof)",
        icon=folium.Icon(color="red", icon="a", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        [LOCATION_B[1], LOCATION_B[0]],
        popup="B: Preysingstr. 53 (Lankwitz)",
        icon=folium.Icon(color="blue", icon="b", prefix="fa"),
    ).add_to(m)

    # ---- Legend ----
    # Build viridis swatches for the in-band commute-time legend
    swatches = "".join(
        f'<span style="display:inline-block; width:18px; height:12px; '
        f'background:{matplotlib.colors.to_hex(plt.cm.viridis(plt.Normalize(15,45)((lo+hi)/2)))};'
        f'"></span>'
        for lo, hi in [(15,20),(20,25),(25,30),(30,35),(35,40),(40,50)]
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px;">
        <b>Commute time difference</b><br>
        <span style="color: #b2182b;">■</span> Faster to <b>A</b> (Adlershof)<br>
        <span style="color: #2166ac;">■</span> Faster to <b>B</b> (Lankwitz)<br>
        <span style="border-bottom: 2px dashed black;">—</span> Equal time<br>
        <span style="border-bottom: 1px dashed gray;">—</span> ±{MAX_DIFF_MINUTES} min band<br>
        <br>
        <b>Inside ±{MAX_DIFF_MINUTES} min band:</b><br>
        Commute time (min):<br>
        {swatches}<br>
        <span style="font-size:11px;">15 &nbsp;&nbsp; 20 &nbsp;&nbsp; 25 &nbsp;&nbsp; 30 &nbsp;&nbsp; 35 &nbsp;&nbsp; 40 &nbsp;&nbsp; 50</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    Geocoder(
        position="topleft",
        collapsed=False,
        add_marker=True,
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CACHE_SCHEMA = 2  # bump when cache layout changes (e.g. added snap arrays)


def _cache_fingerprint():
    """Hash the config that determines query results, for cache invalidation."""
    key = json.dumps({
        "schema": CACHE_SCHEMA,
        "bbox": BERLIN_BBOX,
        "res": GRID_RESOLUTION,
        "A": LOCATION_A,
        "B": LOCATION_B,
    }, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def load_or_query(grid_points, cache_path):
    """Load cached durations+snap if fingerprint matches, else query ORS and save.

    Returns (dur_a, dur_b, snap_a, snap_b). snap arrays are None when loaded
    from an older cache that didn't record them.
    """
    fp = _cache_fingerprint()
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        if str(data["fingerprint"]) == fp:
            print(f"Loaded cached durations from {cache_path.name} (fp={fp})")
            snap_a = data["snap_a"] if "snap_a" in data.files else None
            snap_b = data["snap_b"] if "snap_b" in data.files else None
            if snap_a is None:
                print("  (cache has no snap distances — re-query to enable snap filter)")
            return data["dur_a"], data["dur_b"], snap_a, snap_b
        print(f"Cache fingerprint mismatch ({data['fingerprint']} != {fp}); re-querying")

    n = len(grid_points)
    print(f"\nQuerying travel times to A (Adlershof)...")
    dur_a, snap_a = query_matrix_batched(grid_points, LOCATION_A)
    print(f"  {np.sum(~np.isnan(dur_a))}/{n} valid results")

    print(f"\nQuerying travel times to B (Lankwitz)...")
    dur_b, snap_b = query_matrix_batched(grid_points, LOCATION_B)
    print(f"  {np.sum(~np.isnan(dur_b))}/{n} valid results")

    np.savez(
        cache_path,
        dur_a=dur_a, dur_b=dur_b,
        snap_a=snap_a, snap_b=snap_b,
        fingerprint=fp,
    )
    print(f"Cached durations+snap → {cache_path.name}")
    return dur_a, dur_b, snap_a, snap_b


def main():
    print("Building grid...")
    lons, lats, grid_points = build_grid()
    n = len(grid_points)
    print(f"Grid: {GRID_RESOLUTION}×{GRID_RESOLUTION} = {n} points")

    cache_path = Path(__file__).parent / "ors_cache.npz"
    dur_a, dur_b, snap_a, snap_b = load_or_query(grid_points, cache_path)

    print("\nLoading water polygons for masking...")
    water_cache = Path(__file__).parent / "osm_water.json"
    water_polys = load_or_query_water(water_cache)
    water_mask = build_water_mask(grid_points, water_polys)
    print(f"  {int(water_mask.sum())} grid points in water → NaN")
    dur_a[water_mask] = np.nan
    dur_b[water_mask] = np.nan

    if snap_a is not None:
        snap_bad = (snap_a > MAX_SNAP_DISTANCE_M) | (snap_b > MAX_SNAP_DISTANCE_M)
        print(f"  {int(snap_bad.sum())} grid points snapped >{MAX_SNAP_DISTANCE_M}m → NaN")
        dur_a[snap_bad] = np.nan
        dur_b[snap_bad] = np.nan

    print("\nComputing equal-time contour...")
    lon_fine, lat_fine, lon_mesh, lat_mesh, diff_grid, time_grid = compute_contour(
        lons, lats, dur_a, dur_b
    )

    contour_lines = extract_contour_coords(lon_fine, lat_fine, diff_grid, level=0.0)
    print(f"  Found {len(contour_lines)} contour segments")

    print("\nFetching isochrones for context...")
    iso_a = query_isochrones(LOCATION_A)
    time.sleep(REQUEST_DELAY)
    iso_b = query_isochrones(LOCATION_B)

    print("\nLoading POIs...")
    poi_cache = Path(__file__).parent / "osm_pois.json"
    pois = load_or_query_pois(poi_cache)

    print("\nGeocoding addresses of interest...")
    addr_cache = Path(__file__).parent / "geocode_cache.json"
    addresses = geocode_addresses(ADDRESSES_OF_INTEREST, addr_cache)

    print("\nComputing POI proximity to addresses...")
    prox_cache = Path(__file__).parent / "poi_proximity.json"
    poi_proximity = load_or_compute_poi_proximity(pois, addresses, prox_cache)

    print("\nBuilding map...")
    m = build_map(
        diff_grid, time_grid, lon_fine, lat_fine, contour_lines, iso_a, iso_b,
        pois=pois, addresses=addresses, poi_proximity=poi_proximity,
    )

    out_path = Path(__file__).parent / "equi_commute_berlin.html"
    m.save(str(out_path))
    print(f"\nDone! Map saved to {out_path}")


if __name__ == "__main__":
    main()

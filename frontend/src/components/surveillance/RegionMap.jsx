import { useEffect, useMemo, useRef } from 'react'
import { CircleMarker, GeoJSON, MapContainer, TileLayer, Tooltip } from 'react-leaflet'
import { HOTSPOT_STATUS } from '../../utils/constants'
import { geojsonBounds } from './scale'

/**
 * Base maps for real boundaries. Satellite imagery (Esri World Imagery) with place names on top, or the
 * plain ground. VITE_MAP_TILE_URL replaces the satellite tiles with another provider.
 */
export const BASEMAPS = {
  satellite: {
    url: import.meta.env.VITE_MAP_TILE_URL || 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution:
      import.meta.env.VITE_MAP_TILE_ATTRIBUTION ||
      'Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community',
    labels: import.meta.env.VITE_MAP_TILE_URL
      ? null
      : 'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  },
}

/**
 * Choropleth of regions. Fill shows disease intensity; the outline shows the hotspot status.
 *
 * The view is locked to the loaded regions: it cannot be panned away or zoomed out past them, so a state's
 * districts are shown as that state alone, never as part of a wider (e.g. national) map. Base-map tiles are
 * drawn only for real boundaries and only when VITE_MAP_TILE_URL is set; simulated regions are not a real
 * place, so they always sit on a plain ground.
 */
/** Colour of the "cluster of potential hotspots" layer: a halo around the regions, distinct from the status outlines. */
export const CLUSTER_COLOR = '#c4adff'

export function RegionMap({
  geojson,
  dataKey,
  scale,
  valueKey,
  selectedId,
  onSelect,
  showTiles,
  watermark,
  showClusters = false,
  hospitals = [],
  basemap = 'satellite',
}) {
  const bounds = useMemo(() => geojsonBounds(geojson), [geojson])
  const tiles = Boolean(showTiles && basemap === 'satellite')
  const layer = BASEMAPS.satellite
  const selectRef = useRef(onSelect)
  useEffect(() => {
    selectRef.current = onSelect
  }, [onSelect])

  const style = useMemo(
    () => (feature) => {
      const { status, region_id: regionId } = feature.properties
      const selected = regionId === selectedId
      const flagged = status === 'CANDIDATE_HOTSPOT' || status === 'ELEVATED'
      return {
        fillColor: scale.color(feature.properties[valueKey]),
        fillOpacity: tiles ? 0.45 : 1,
        color: selected ? '#ffffff' : flagged ? HOTSPOT_STATUS[status].color : tiles ? 'rgba(255,255,255,0.55)' : '#12233c',
        weight: selected ? 3.5 : status === 'CANDIDATE_HOTSPOT' ? 3 : status === 'ELEVATED' ? 2.5 : 1,
        dashArray: status === 'ELEVATED' && !selected ? '6 4' : undefined,
      }
    },
    [scale, valueKey, selectedId, tiles],
  )

  const clustered = useMemo(
    () => ({ ...geojson, features: geojson.features.filter((f) => f.properties.potential_hotspot_cluster) }),
    [geojson],
  )

  const onEachFeature = (feature, layer) => {
    const { region_name: name, status_label: label, potential_hotspot_cluster: cluster } = feature.properties
    const clusterText = showClusters && cluster ? ` (cluster of potential hotspots ${cluster})` : ''
    layer.bindTooltip(`${name}: ${label}${clusterText}`, { sticky: true })
    layer.on({ click: () => selectRef.current(feature.properties.region_id) })
  }

  return (
    <div
      role="group"
      aria-label="Map of regions coloured by disease intensity. The table below lists the same regions."
      className="relative isolate"
    >
      <MapContainer
        key={`${tiles ? 'tiles' : 'plain'}:${bounds.flat().join(',')}`}
        bounds={bounds}
        boundsOptions={{ padding: [12, 12] }}
        maxBounds={padBounds(bounds, 0.15)}
        maxBoundsViscosity={1}
        scrollWheelZoom={false}
        zoomSnap={0.25}
        attributionControl={tiles}
        whenReady={(event) => event.target.setMinZoom(event.target.getZoom())}
        className="h-[26rem] w-full"
      >
        {tiles ? <TileLayer url={layer.url} attribution={layer.attribution} maxNativeZoom={17} /> : null}
        {/* Flagged regions are drawn last so their outlines sit on top of their neighbours'. */}
        <GeoJSON key={`${dataKey}:${showClusters}`} data={orderForDrawing(geojson)} style={style} onEachFeature={onEachFeature} />
        {/* Extra layer: a translucent halo around regions in a cluster of potential hotspots. Not clickable. */}
        {showClusters && clustered.features.length ? (
          <GeoJSON
            key={`${dataKey}:clusters`}
            data={clustered}
            interactive={false}
            style={() => ({ fill: false, color: CLUSTER_COLOR, weight: 9, opacity: 0.38, lineJoin: 'round' })}
          />
        ) : null}
        {tiles && layer.labels ? <TileLayer url={layer.labels} pane="shadowPane" opacity={0.85} /> : null}
        {/* Hospital sites: their own location, or the district centre (dashed ring) when none was given. */}
        {hospitals.map((hospital) => (
          <CircleMarker
            key={hospital.id}
            center={[hospital.latitude, hospital.longitude]}
            radius={hospital.location === 'exact' ? 7 : 6}
            pathOptions={{
              color: '#ffffff',
              weight: 2,
              dashArray: hospital.location === 'exact' ? undefined : '3 3',
              fillColor: hospital.online ? '#5fd3a0' : '#8b98ad',
              fillOpacity: 1,
            }}
          >
            <Tooltip direction="top" offset={[0, -6]}>
              <span className="font-semibold">{hospital.name}</span>
              <br />
              {hospital.online ? 'Online' : 'Offline'}, {hospital.region_name}
              <br />
              {hospital.location === 'exact' ? 'Hospital location' : 'District centre (exact site not given)'}
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
      {watermark ? (
        // Part of the map itself, above the regions and not dismissable; clicks pass through to the map.
        <div className="pointer-events-none absolute inset-0 z-[500] flex items-center justify-center overflow-hidden rounded-lg">
          <p
            data-testid="map-watermark"
            className="-rotate-12 select-none whitespace-nowrap rounded border-2 border-alert/60 bg-surface/70 px-4 py-2 text-lg font-bold uppercase tracking-wider text-alert sm:text-2xl"
          >
            {watermark}
          </p>
        </div>
      ) : null}
    </div>
  )
}

/** [[south, west], [north, east]] grown by ``share`` of its size on every side. */
export function padBounds([[south, west], [north, east]], share) {
  const dLat = (north - south) * share
  const dLon = (east - west) * share
  return [
    [south - dLat, west - dLon],
    [north + dLat, east + dLon],
  ]
}

const DRAW_ORDER = { INSUFFICIENT_DATA: 0, NORMAL: 0, ELEVATED: 1, CANDIDATE_HOTSPOT: 2 }

function orderForDrawing(geojson) {
  const features = [...geojson.features].sort((a, b) => (DRAW_ORDER[a.properties.status] ?? 0) - (DRAW_ORDER[b.properties.status] ?? 0))
  return { ...geojson, features }
}

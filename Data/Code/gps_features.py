import numpy as np

from .geo import haversine_meters

GPS_COLS = [
    "gps_total_haversine_meters",
    "gps_number_of_stay_points",
    "gps_total_time_at_clusters_seconds",
    "gps_location_entropy",
    "gps_location_variance",
    "gps_radius_of_gyration_meters",
]


def simulate_gps_features(home_lat, home_lon, mobility_level, awake_minutes, rng):
    """Simulate one participant-day's GPS itinerary and derive trip stats from it."""
    k = int(np.clip(rng.poisson(2 + 10 * mobility_level), 1, 20))

    spread_deg = 0.0003 + 0.006 * mobility_level
    lat_pts = np.concatenate([[home_lat], home_lat + rng.normal(0, spread_deg, k)])
    lon_pts = np.concatenate([[home_lon], home_lon + rng.normal(0, spread_deg, k)])

    # dwell-time share at each location (including home)
    dwell_frac = rng.dirichlet(np.ones(k + 1))
    time_at_clusters_seconds = 0.75 * awake_minutes * 60 * rng.uniform(0.8, 1.0)

    entropy = -np.sum(dwell_frac * np.log2(dwell_frac + 1e-12))

    centroid_lat = np.average(lat_pts, weights=dwell_frac)
    centroid_lon = np.average(lon_pts, weights=dwell_frac)
    dists_from_centroid = haversine_meters(lat_pts, lon_pts, centroid_lat, centroid_lon)

    radius_of_gyration = np.sqrt(np.sum(dwell_frac * dists_from_centroid ** 2))
    location_variance = np.average(dists_from_centroid ** 2, weights=dwell_frac)

    # path length visiting points in order, home -> stop1 -> ... -> home
    loop_lat = np.concatenate([lat_pts, [home_lat]])
    loop_lon = np.concatenate([lon_pts, [home_lon]])
    leg_dists = haversine_meters(loop_lat[:-1], loop_lon[:-1], loop_lat[1:], loop_lon[1:])
    total_haversine = np.sum(leg_dists)

    return {
        "gps_total_haversine_meters": np.clip(total_haversine, 0, 20000),
        "gps_number_of_stay_points": k,
        "gps_total_time_at_clusters_seconds": np.clip(time_at_clusters_seconds, 0, 100000),
        "gps_location_entropy": np.clip(entropy, 0, 10),
        "gps_location_variance": np.clip(location_variance, 0, 500000),
        "gps_radius_of_gyration_meters": np.clip(radius_of_gyration, 0, 20000),
    }


"""
Explainable AI module: Provides human-readable explanations for habitability predictions.
"""
import numpy as np
import pandas as pd


def check_in_habitable_zone(row):
    """
    Checks if planet is in habitable zone based on stellar flux.
    """
    flux = row['star_luminosity'] / (row['orbit_semi_major_axis']**2)
    return 0.2 < flux < 2.2, flux


def check_earth_size(row):
    """
    Checks if planet has Earth-like radius.
    """
    return 0.5 < row['planet_radius'] < 2.0, row['planet_radius']


def check_stable_orbit(row, system_df=None):
    """
    Check if orbit is likely stable (simplified rule).
    """
    # Check orbital eccentricity proxy (we don't have eccentricity, use a default value)
    eccentricity = row.get('pl_eccen', 0.0)
    is_stable = eccentricity < 0.5  # Simplified: e < 0.5 considered stable
    return is_stable, eccentricity


def check_stellar_type(row):
    """
    Check if star type is suitable.
    """
    temp = row['star_temp_k']
    if 3700 < temp < 7200:  # K, G, F type stars
        return True, temp
    return False, temp


def check_stellar_luminosity(row):
    """
    Check if stellar luminosity is suitable.
    """
    lum = row['star_luminosity']
    if 0.01 < lum < 10:  # Not too dim, not too bright
        return True, lum
    return False, lum


def generate_habitability_explanation(row):
    """
    Generates a human-readable explanation of habitability.
    """
    planet_name = row.get('planet_name', 'This Planet')
    hostname = row.get('hostname', 'Its Star')
    
    explanation = []
    checks_passed = []
    checks_failed = []
    
    # HZ Check
    in_hz, flux = check_in_habitable_zone(row)
    if in_hz:
        checks_passed.append(f"✅ Located **inside the Habitable Zone** (Flux: {flux:.2f}x Earth)")
    else:
        checks_failed.append(f"❌ Located **outside the Habitable Zone** (Flux: {flux:.2f}x Earth)")
    
    # Planet Radius
    earth_size, rad = check_earth_size(row)
    if earth_size:
        checks_passed.append(f"✅ Earth-like size ({rad:.2f} R_earth)")
    else:
        checks_failed.append(f"❌ Non-Earth-like size ({rad:.2f} R_earth)")
    
    # Star Temperature
    star_ok, temp = check_stellar_type(row)
    if star_ok:
        checks_passed.append(f"✅ Suitable star temperature ({temp:.0f} K)")
    else:
        checks_failed.append(f"❌ Star temperature too extreme ({temp:.0f} K)")
    
    # Luminosity Check
    lum_ok, lum = check_stellar_luminosity(row)
    if lum_ok:
        checks_passed.append(f"✅ Suitable stellar luminosity ({lum:.2f} L_sun)")
    else:
        checks_failed.append(f"❌ Extreme stellar luminosity ({lum:.2f} L_sun)")
    
    # Stable orbit (simplified)
    stable, _ = check_stable_orbit(row)
    if stable:
        checks_passed.append("✅ Orbit likely stable")
    else:
        checks_failed.append("❌ Orbit potentially unstable")
    
    explanation_text = f"## Habitability Analysis for **{planet_name}** (Host star: {hostname})\n\n"
    
    if checks_passed:
        explanation_text += "### Key strengths:\n"
        for check in checks_passed:
            explanation_text += f"- {check}\n"
        explanation_text += "\n"
    
    if checks_failed:
        explanation_text += "### Key challenges:\n"
        for check in checks_failed:
            explanation_text += f"- {check}\n"
        explanation_text += "\n"
    
    # Overall summary
    overall_score = row.get('overall_habitability_score', None)
    if overall_score is not None:
        explanation_text += f"### Overall Habitability Score: **{overall_score:.1%}**\n"
    
    return explanation_text, checks_passed, checks_failed


def get_explanation_dict(row):
    """
    Returns a structured dictionary of habitability checks.
    """
    checks = {}
    in_hz, flux = check_in_habitable_zone(row)
    checks['in_habitable_zone'] = in_hz
    checks['stellar_flux'] = flux
    
    earth_size, rad = check_earth_size(row)
    checks['earth_like_radius'] = earth_size
    checks['planet_radius'] = rad
    
    star_ok, temp = check_stellar_type(row)
    checks['suitable_star_temp'] = star_ok
    checks['star_temp'] = temp
    
    lum_ok, lum = check_stellar_luminosity(row)
    checks['suitable_luminosity'] = lum_ok
    checks['star_luminosity'] = lum
    
    stable, ecc = check_stable_orbit(row)
    checks['likely_stable_orbit'] = stable
    checks['eccentricity'] = ecc
    
    return checks

"""Census-region assignment for U.S. MSAs.

Used throughout this project's analysis notebooks to aggregate MSA-level
structural gap estimates into the four standard Census Bureau regions.
"""

NORTHEAST = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}
MIDWEST = {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"}
SOUTH = {
    "DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
    "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX",
}
WEST = {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"}

REGIONS = ["Northeast", "Midwest", "South", "West"]

STATE_TO_REGION = {}
for _s in NORTHEAST:
    STATE_TO_REGION[_s] = "Northeast"
for _s in MIDWEST:
    STATE_TO_REGION[_s] = "Midwest"
for _s in SOUTH:
    STATE_TO_REGION[_s] = "South"
for _s in WEST:
    STATE_TO_REGION[_s] = "West"


def get_primary_state(msa_name: str) -> str | None:
    """Return the first two-letter state code listed in an MSA name.

    This is a simplification for multi-state MSAs (e.g. "New York-Newark-
    Jersey City, NY-NJ-PA" -> "NY"), not an official Census rule, but it is
    the convention used consistently across this project.
    """
    if not isinstance(msa_name, str) or "," not in msa_name:
        return None
    state_part = msa_name.split(",")[-1].strip().replace("/", "-")
    primary = state_part.split("-")[0].strip()
    return primary if len(primary) == 2 else None


def assign_region(msa_name: str) -> str | None:
    """Map an MSA name to one of the four Census regions, or None if the
    primary state can't be parsed or isn't recognized."""
    return STATE_TO_REGION.get(get_primary_state(msa_name))


def add_region_column(df, msa_col: str = "MSA_Name", region_col: str = "Region"):
    """Return a copy of df with a Region column added, warning (not raising)
    on any MSAs that couldn't be assigned."""
    out = df.copy()
    out[region_col] = out[msa_col].map(assign_region)
    unmatched = out.loc[out[region_col].isna(), msa_col].unique()
    if len(unmatched) > 0:
        print(f"WARNING -- {len(unmatched)} MSAs unassigned to a region: {list(unmatched)}")
    return out

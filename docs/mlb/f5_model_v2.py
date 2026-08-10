import statsapi
print(statsapi.get("person", {"personId": 592450, "hydrate":
    "stats(group=[hitting],type=[sabermetrics],season=2026)"}))
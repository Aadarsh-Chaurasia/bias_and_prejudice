"""
Deterministic synthetic S1/S2/S3 + ground truth with the same schema as the competition data:
  source files: entity_id, business_name, business_address, country
  ground truth: source1_entity_id, matched_entity_ids (comma-separated S2/S3 ids)

Used only when the real sample (data/sample/*.tsv from generate_sample.py) is not available.
Noise model mirrors what the notebooks show: Pvt Ltd / Private Limited variants, reordered
address parts, dropped PIN, typos, abbreviations (Road/Rd), casing, accents (France).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_IN_STATES = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubli", "Mangaluru"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Gwalior", "Jabalpur"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Haryana": ["Faridabad", "Gurugram", "Ballabgarh", "Panipat"],
    "Kerala": ["Kochi", "Thrissur", "Kozhikode", "Ernakulam"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Agra", "Varanasi"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
}
_US_STATES = {
    "CA": ["Los Angeles", "San Diego", "San Jose", "Fresno"],
    "TX": ["Houston", "Dallas", "Austin", "El Paso"],
    "NY": ["New York", "Buffalo", "Rochester", "Albany"],
    "AZ": ["Phoenix", "Flagstaff", "Tucson", "Mesa"],
    "FL": ["Miami", "Orlando", "Tampa", "Jacksonville"],
    "IL": ["Chicago", "Aurora", "Naperville", "Peoria"],
}
_FR_CITIES = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Montpellier", "Rennes"]

_IN_WORDS = ["Shree", "Sai", "Ganesh", "Lakshmi", "Balaji", "Krishna", "Om", "Sri", "Durga", "Ambika",
             "Bharat", "Indo", "Global", "National", "Royal", "Star", "Sagar", "Vijay", "Kaveri", "Sagar",
             "Satyanarayan", "Jyoti", "Mahalaxmi", "Annapurna", "Hari", "Anand", "Raj", "Maruti", "Surya",
             "Aditya", "Ashok", "Gopal", "Vishnu", "Tirupati", "Jai", "Mangal", "Kalyan", "Navkar", "Pooja"]
_IN_TRADE = ["Traders", "Textiles", "Enterprises", "Industries", "Impex", "Exports", "Agencies", "Pharma",
             "Steels", "Plastics", "Foods", "Electricals", "Motors", "Builders", "Chemicals", "Silk", "Sago",
             "Hardware", "Jewellers", "Logistics", "Forging", "Utility", "Services", "Solutions"]
_IN_SUFFIX = ["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "Ltd", "", "", "LLP"]
_US_WORDS = ["Eagle", "Summit", "Pioneer", "Liberty", "Canyon", "Harbor", "Golden", "Silver", "Maple",
             "Oak", "River", "Lakeside", "Pacific", "Atlantic", "Frontier", "Sunrise", "Bluebird", "Redwood",
             "Abad", "Keystone", "Granite", "Evergreen", "Horizon", "Prairie", "Cedar", "Falcon", "Mesa"]
_US_TRADE = ["Partnership", "Holdings", "Consulting", "Logistics", "Dental", "Realty", "Plumbing",
             "Auto Repair", "Bakery", "Construction", "Insurance", "Law Group", "Capital", "Medical",
             "Roofing", "Landscaping", "Software", "Cleaning", "Market", "Supply"]
_US_SUFFIX = ["Inc", "Incorporated", "LLC", "Corp", "Corporation", "Co", ""]
_US_STREETS = ["Main", "Oak", "Maple", "Indian", "Washington", "Lake", "Hill", "Park", "Cedar", "Elm",
               "Sunset", "Pine", "Ridge", "Mill", "Church", "Highland"]
_US_STYPE = [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Drive", "Dr"), ("Boulevard", "Blvd"),
             ("Lane", "Ln")]
_IN_LOCAL = ["Rajiv Colony", "Gandhi Nagar", "MG Road", "Station Road", "Wilson Garden", "Civil Lines",
             "Sector 15", "Industrial Area", "Shivaji Nagar", "Nehru Place", "Anna Nagar", "Bhavani Mansion",
             "Laxmi Vihar", "Old City", "Market Yard", "Model Town", "Patel Chowk", "Hill View Enclave"]
_FR_WORDS = ["Societe", "Boulangerie", "Cafe", "Atelier", "Maison", "Garage", "Pharmacie", "Cabinet",
             "Groupe", "Etablissements"]
_FR_NAMES = ["Dupont", "Lefevre", "Moreau", "Gerard", "Benoit", "Lemaitre", "Chevalier", "Rousseau",
             "Fontaine", "Mercier", "Leclerc", "Brassens"]
_FR_STREETS = ["Rue de la Paix", "Avenue des Champs", "Boulevard Saint-Germain", "Rue Victor Hugo",
               "Chemin des Vignes", "Allee des Tilleuls", "Place de l'Eglise", "Rue Pasteur"]
_ACCENT = {"e": "é", "a": "à", "c": "ç", "o": "ô", "u": "ù", "i": "î"}


def _typo(rng: np.random.Generator, s: str) -> str:
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 1))
    op = rng.integers(0, 4)
    if op == 0:
        return s[:i] + s[i + 1:]
    if op == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == 2:
        return s[:i] + chr(int(rng.integers(97, 123))) + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def _base_entity(rng: np.random.Generator, country: str, pins: dict) -> dict:
    if country == "India":
        state = rng.choice(list(_IN_STATES))
        city = rng.choice(_IN_STATES[state])
        words = rng.choice(_IN_WORDS, size=int(rng.integers(1, 3)), replace=False).tolist()
        name = " ".join(words + [rng.choice(_IN_TRADE)])
        suffix = rng.choice(_IN_SUFFIX)
        pin = rng.choice(pins[("India", city)])
        local = rng.choice(_IN_LOCAL)
        num = f"{rng.integers(1, 999)}/{rng.integers(1, 99)}" if rng.random() < .5 else f"Office No. {rng.integers(1, 600)}-A"
        parts = [num, local, city, state]
        return dict(name=name, suffix=suffix, parts=parts, pin=pin, country=country)
    if country == "US":
        st = rng.choice(list(_US_STATES))
        city = rng.choice(_US_STATES[st])
        name = f"{rng.choice(_US_WORDS)} {rng.choice(_US_TRADE)}"
        suffix = rng.choice(_US_SUFFIX)
        stype = _US_STYPE[int(rng.integers(len(_US_STYPE)))]
        street = f"{rng.integers(10, 9999)} {rng.choice(_US_STREETS)} {stype[0]}"
        pin = rng.choice(pins[("US", city)])
        return dict(name=name, suffix=suffix, parts=[street, city, st], pin=pin, country=country, stype=stype)
    city = rng.choice(_FR_CITIES)
    name = f"{rng.choice(_FR_WORDS)} {rng.choice(_FR_NAMES)}"
    street = f"{rng.integers(1, 200)} {rng.choice(_FR_STREETS)}"
    pin = rng.choice(pins[("France", city)])
    return dict(name=name, suffix=rng.choice(["SARL", "SAS", ""]), parts=[street, city], pin=pin, country=country)


def _render(rng: np.random.Generator, e: dict, noisy: bool) -> tuple[str, str, str]:
    name, suffix, parts, pin = e["name"], e["suffix"], list(e["parts"]), e["pin"]
    if noisy:
        if e["country"] == "India" and rng.random() < .5:
            suffix = rng.choice(_IN_SUFFIX)
        if e["country"] == "US" and rng.random() < .4:
            suffix = rng.choice(_US_SUFFIX)
        if rng.random() < .3:
            name = _typo(rng, name)
        if rng.random() < .15:
            name = name.upper()
        if rng.random() < .1:
            w = name.split()
            if len(w) > 2:
                name = " ".join(w[1:])
        if e["country"] == "US" and rng.random() < .5:
            full, short = e["stype"]
            parts[0] = parts[0].replace(full, short)
        if e["country"] == "France" and rng.random() < .5:
            name = "".join(_ACCENT.get(c, c) if rng.random() < .15 else c for c in name)
        if rng.random() < .25:
            pin = None
        if rng.random() < .2:                    # reorder address parts
            rng.shuffle(parts)
        if rng.random() < .15 and len(parts) > 2:
            parts = parts[1:]
        if rng.random() < .1:
            parts[-1] = _typo(rng, parts[-1])
    full_name = f"{name} {suffix}".strip()
    addr = ", ".join(parts)
    if pin:
        addr = f"{addr} {pin}" if e["country"] != "France" else f"{parts[0]}, {pin} {', '.join(parts[1:])}"
    return full_name, addr, e["country"]


def make_synthetic(n_s1: int = 20_000, n_s2: int = 45_000, n_s3: int = 47_000, match_rate: float = .6,
                   seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    countries = np.array(["India", "US", "France"])
    probs = [.42, .5, .08]
    pins: dict = {}
    for st, cities in _IN_STATES.items():
        for c in cities:
            pins[("India", c)] = [str(rng.integers(110000, 855000)) for _ in range(6)]
    for st, cities in _US_STATES.items():
        for c in cities:
            pins[("US", c)] = [f"{rng.integers(10000, 99999)}" for _ in range(6)]
    for c in _FR_CITIES:
        pins[("France", c)] = [f"{rng.integers(10, 95):02d}{rng.integers(0, 999):03d}" for _ in range(4)]

    def ids(prefix: str, n: int) -> list[str]:
        return [f"{prefix}-{x}" for x in rng.choice(900_000_000, size=n, replace=False) + 100_000_000]

    s1_ids, s2_ids, s3_ids = ids("S1", n_s1), ids("S2", n_s2), ids("S3", n_s3)
    rows = {1: [], 2: [], 3: []}
    gt = []
    s2_ptr = s3_ptr = 0
    for i in range(n_s1):
        e = _base_entity(rng, rng.choice(countries, p=probs), pins)
        rows[1].append((s1_ids[i], *_render(rng, e, noisy=False)))
        matches = []
        if rng.random() < match_rate:
            r = rng.random()
            targets = [2] if r < .4 else [3] if r < .8 else [2, 3]
            for t in targets:
                if t == 2 and s2_ptr < n_s2:
                    rows[2].append((s2_ids[s2_ptr], *_render(rng, e, noisy=True)))
                    matches.append(s2_ids[s2_ptr])
                    s2_ptr += 1
                elif t == 3 and s3_ptr < n_s3:
                    rows[3].append((s3_ids[s3_ptr], *_render(rng, e, noisy=True)))
                    matches.append(s3_ids[s3_ptr])
                    s3_ptr += 1
        gt.append((s1_ids[i], ",".join(matches) if matches else None))
    for ptr, n, t, idl in ((s2_ptr, n_s2, 2, s2_ids), (s3_ptr, n_s3, 3, s3_ids)):
        for j in range(ptr, n):
            e = _base_entity(rng, rng.choice(countries, p=probs), pins)
            rows[t].append((idl[j], *_render(rng, e, noisy=True)))

    cols = ["entity_id", "business_name", "business_address", "country"]
    frames = []
    for t in (1, 2, 3):
        df = pd.DataFrame(rows[t], columns=cols).sample(frac=1, random_state=seed).reset_index(drop=True)
        # a few missing values like the real data
        m = rng.random(len(df)) < .003
        df.loc[m, "business_address"] = np.nan
        frames.append(df)
    g = pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"])
    return frames[0], frames[1], frames[2], g

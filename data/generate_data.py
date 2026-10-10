"""Synthetic point-of-sale data for a Baku pharmacy chain (Jan-Sep 2026).

Baskets are generated from latent shopping "missions" (cold & flu, baby care,
chronic medication, ...) whose weights change by season, so the data contains
real co-purchase structure, seasonal rules and substitute products.
A handful of data-quality issues (returns, duplicate scan lines) are injected on purpose.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- products
PRODUCTS = [
    # id, name, category, price AZN, rx
    ("P01", "Cold & flu powder", "Cold & flu", 6.40, 0),
    ("P02", "Throat lozenges", "Cold & flu", 4.20, 0),
    ("P03", "Nasal spray", "Cold & flu", 5.80, 0),
    ("P04", "Cough syrup", "Cold & flu", 7.90, 0),
    ("P05", "Vitamin C 1000", "Vitamins", 8.50, 0),
    ("P06", "Paper tissues", "Hygiene", 1.20, 0),
    ("P07", "Digital thermometer", "Devices", 12.00, 0),
    ("P08", "Ibuprofen 400", "Pain relief", 3.60, 0),
    ("P09", "Paracetamol 500", "Pain relief", 2.40, 0),
    ("P10", "Metamizole 500", "Pain relief", 2.10, 0),
    ("P11", "Pain relief gel", "Pain relief", 9.40, 0),
    ("P12", "Amoxicillin 500", "Antibiotics (Rx)", 7.20, 1),
    ("P13", "Azithromycin 500", "Antibiotics (Rx)", 11.50, 1),
    ("P14", "Probiotic capsules", "Digestive", 14.80, 0),
    ("P15", "Antacid tablets", "Digestive", 5.30, 0),
    ("P16", "Activated charcoal", "Digestive", 1.10, 0),
    ("P17", "Oral rehydration salts", "Digestive", 2.90, 0),
    ("P18", "Loperamide 2", "Digestive", 3.20, 0),
    ("P19", "Amlodipine 5", "Cardio (Rx)", 6.60, 1),
    ("P20", "Lisinopril 10", "Cardio (Rx)", 5.90, 1),
    ("P21", "Blood pressure monitor", "Devices", 58.00, 0),
    ("P22", "Magnesium B6", "Vitamins", 9.80, 0),
    ("P23", "Omega-3", "Vitamins", 16.50, 0),
    ("P24", "Metformin 850", "Diabetes (Rx)", 4.70, 1),
    ("P25", "Glucose test strips", "Diabetes", 24.00, 0),
    ("P26", "Lancets", "Diabetes", 7.50, 0),
    ("P27", "Infant formula", "Baby care", 21.00, 0),
    ("P28", "Diapers", "Baby care", 18.50, 0),
    ("P29", "Baby wipes", "Baby care", 3.40, 0),
    ("P30", "Diaper cream", "Baby care", 8.20, 0),
    ("P31", "Sunscreen SPF50", "Skin & sun", 19.90, 0),
    ("P32", "After-sun gel", "Skin & sun", 12.40, 0),
    ("P33", "Insect repellent", "Skin & sun", 6.90, 0),
    ("P34", "Moisturising cream", "Skin & sun", 11.20, 0),
    ("P35", "Plasters", "First aid", 2.30, 0),
    ("P36", "Antiseptic solution", "First aid", 2.80, 0),
    ("P37", "Elastic bandage", "First aid", 3.90, 0),
    ("P38", "Cotton pads", "Hygiene", 1.80, 0),
    ("P39", "Toothpaste", "Dental", 4.60, 0),
    ("P40", "Toothbrush", "Dental", 3.50, 0),
    ("P41", "Mouthwash", "Dental", 6.30, 0),
    ("P42", "Hand sanitiser", "Hygiene", 2.60, 0),
    ("P43", "Contact lens solution", "Eye care", 13.60, 0),
    ("P44", "Eye drops", "Eye care", 7.40, 0),
    ("P45", "Antihistamine (cetirizine)", "Allergy", 4.90, 0),
    ("P46", "Multivitamin", "Vitamins", 15.20, 0),
    ("P47", "Vitamin D3", "Vitamins", 10.60, 0),
    ("P48", "Zinc", "Vitamins", 7.80, 0),
    ("P49", "Iron supplement", "Vitamins", 9.10, 0),
]
products = pd.DataFrame(PRODUCTS, columns=["product_id", "product_name", "category", "list_price", "is_rx"])
PRICE = dict(zip(products.product_id, products.list_price))


def season(month, winter=1.0, spring=1.0, summer=1.0, autumn=1.0):
    return {12: winter, 1: winter, 2: winter, 3: spring, 4: spring, 5: spring,
            6: summer, 7: summer, 8: summer, 9: autumn, 10: autumn, 11: autumn}[month]


# mission -> (seasonal weight fn, list of (items or exclusive choice, prob))
# an element ("ONE", [(pid, w), ...], p) means "with prob p buy exactly one of these"
MISSIONS = {
    "cold_flu": (lambda m: 2.6 * season(m, 1.0, 0.55, 0.15, 0.45) * (1.25 if m in (1, 2) else 1.0),
                 [("P01", .55), ("P02", .45), ("P03", .40), ("P04", .30), ("P05", .35), ("P06", .30),
                  ("ONE", [("P09", .7), ("P08", .2), ("P10", .1)], .40), ("P07", .05)]),
    "pain": (lambda m: 1.6,
             [("ONE", [("P08", .45), ("P09", .35), ("P10", .20)], 1.0), ("P11", .18)]),
    "antibiotic": (lambda m: 0.75 * season(m, 1.4, 1.0, 0.7, 1.0),
                   [("ONE", [("P12", .6), ("P13", .4)], 1.0), ("P14", .55), ("P02", .10)]),
    "digestive": (lambda m: 1.0 * season(m, 0.9, 1.0, 1.5, 1.0),
                  [("P15", .45), ("P16", .35), ("P17", .25), ("P18", .30), ("P14", .12)]),
    "cardio": (lambda m: 1.3,
               [("P19", .60), ("P20", .50), ("P22", .28), ("P23", .18), ("P21", .04)]),
    "diabetes": (lambda m: 0.55,
                 [("P24", .85), ("P25", .55), ("P26", .40)]),
    "baby": (lambda m: 0.9,
             [("P27", .45), ("P28", .70), ("P29", .55), ("P30", .20)]),
    "sun": (lambda m: 1.8 * season(m, 0.0, 0.12, 1.0, 0.25),
            [("P31", .80), ("P32", .35), ("P33", .30)]),
    "first_aid": (lambda m: 0.7 * season(m, 0.8, 1.0, 1.3, 1.0),
                  [("P35", .70), ("P36", .50), ("P37", .22), ("P38", .25)]),
    "dental_hygiene": (lambda m: 1.1,
                       [("P39", .70), ("P40", .45), ("P41", .28), ("P42", .15), ("P38", .10)]),
    "eye": (lambda m: 0.45,
            [("P43", .65), ("P44", .45)]),
    "allergy": (lambda m: 1.5 * season(m, 0.15, 1.0, 0.45, 0.35) * (1.4 if m in (4, 5) else 1.0),
                [("P45", .90), ("P44", .30), ("P03", .25), ("P06", .20)]),
    "vitamins": (lambda m: 0.95 * season(m, 1.4, 1.0, 0.7, 1.0),
                 [("P46", .40), ("P47", .35), ("P48", .20), ("P49", .22), ("P23", .18), ("P05", .20)]),
    "skin": (lambda m: 0.6,
             [("P34", .65), ("P38", .35), ("P31", .05)]),
}
IMPULSE = ["P06", "P42", "P02", "P35", "P16", "P38", "P39", "P09", "P05"]
IMPULSE_W = np.array([5, 3, 3, 2, 2, 2, 1, 2, 1], float)
IMPULSE_W /= IMPULSE_W.sum()

STORES = ["Nasimi", "Yasamal", "Narimanov", "Khatai", "Binagadi", "Sabail"]
STORE_W = np.array([1.35, 1.15, 1.0, 1.0, 0.8, 0.7]); STORE_W /= STORE_W.sum()
HOURS = np.arange(8, 24)
HOUR_W = np.array([3, 6, 8, 8, 7, 7, 6, 6, 6, 7, 9, 10, 9, 7, 4, 2], float); HOUR_W /= HOUR_W.sum()


def draw_mission(name):
    items = set()
    for el in MISSIONS[name][1]:
        if el[0] == "ONE":
            _, choices, p = el
            if rng.random() < p:
                ids, w = zip(*choices)
                items.add(rng.choice(ids, p=np.array(w) / sum(w)))
        else:
            pid, p = el
            if rng.random() < p:
                items.add(pid)
    return items


lines, baskets = [], []
days = pd.date_range("2026-01-01", "2026-09-30", freq="D")
bid = 0
for day in days:
    names = list(MISSIONS)
    w = np.array([MISSIONS[n][0](day.month) for n in names]); w /= w.sum()
    n_baskets = rng.poisson(165 * (1.12 if day.dayofweek in (0, 5) else 1.0) * (1.15 if day.month in (1, 2) else 1.0))
    for _ in range(n_baskets):
        bid += 1
        k = rng.choice([1, 2, 3], p=[.72, .24, .04])
        chosen = rng.choice(names, size=k, replace=False, p=w)
        items = set()
        for mname in chosen:
            items |= draw_mission(mname)
        if not items or rng.random() < 0.22:
            items.add(rng.choice(IMPULSE, p=IMPULSE_W))
        ts = day + pd.Timedelta(hours=int(rng.choice(HOURS, p=HOUR_W)), minutes=int(rng.integers(0, 60)))
        store = rng.choice(STORES, p=STORE_W)
        loyalty = rng.random() < 0.38
        baskets.append((f"B{bid:06d}", ts, store, int(loyalty)))
        for pid in sorted(items):
            qty = 1 if rng.random() < 0.86 else 2
            price = round(PRICE[pid] * (0.9 if rng.random() < 0.08 else 1.0), 2)
            lines.append((f"B{bid:06d}", pid, qty, price))

baskets = pd.DataFrame(baskets, columns=["basket_id", "ts", "store", "loyalty_card"])
lines = pd.DataFrame(lines, columns=["basket_id", "product_id", "qty", "unit_price"])

# ---- inject data-quality issues --------------------------------------------
ret = lines.sample(180, random_state=SEED).copy()          # returns: negative qty lines
ret["qty"] = -ret["qty"]
dup = lines.sample(95, random_state=SEED + 1)               # double-scanned lines
lines = pd.concat([lines, ret, dup], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

products.to_csv(OUT / "products.csv", index=False)
baskets.to_csv(OUT / "baskets.csv", index=False)
lines.to_csv(OUT / "basket_lines.csv", index=False)
print(f"baskets={len(baskets):,} lines={len(lines):,} products={len(products)}")

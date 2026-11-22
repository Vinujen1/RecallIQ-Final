"""Exact-title groups and initial (heuristic) product-family assignment.

Product families are a starting point for later phases, NOT ground truth:
- "high"   : taken from the record's own specific official category.
- "medium" : several specific official categories, or exactly one keyword match.
- "low"    : no match, or keywords pointing at several families.
Every record that is not "high" gets needs_review = True and a review_reason.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

# Official categories too broad to identify a product family on their own.
GENERIC_CATEGORIES = frozenset({"Food", "Other", "Multiple food items", "Consumer products"})

# Family names reuse the dataset's own category vocabulary so families stay comparable
# with officially categorized records.
FOOD_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Dairy": ("cheese", "yogurt", "yoghurt", "kefir", "paneer", "ghee", "milk", "ice cream"),
    "Meat and poultry": (
        "beef", "pork", "chicken", "turkey", "sausage", "ham", "bacon", "veal", "lamb",
        "meat", "salami", "pepperoni", "jerky", "burger",
    ),
    "Fish and seafood": (
        "fish", "salmon", "tuna", "shrimp", "oyster", "clam", "mussel", "crab", "lobster",
        "cod", "sardine", "anchovy", "anchovies", "scallop", "seafood", "mackerel", "herring",
    ),
    "Fruits and vegetables": (
        "fruit", "vegetable", "apple", "berry", "berries", "mango", "lettuce", "spinach",
        "sprout", "mushroom", "onion", "carrot", "tomato", "peach", "peaches", "melon",
        "cucumber", "salad", "cantaloupe", "kale", "cilantro", "strawberry", "strawberries",
        "asparagus",
    ),
    "Bread": ("bread", "bun", "bagel", "pita", "tortilla"),
    "Grain products": ("flour", "pasta", "noodle", "cereal", "rice", "oat", "oatmeal", "granola"),
    "Nuts, grains, and seeds": (
        "nut", "almond", "cashew", "pistachio", "peanut", "walnut", "pecan", "hazelnut",
        "seed", "tahini",
    ),
    "Candy, confectionary, snacks and sweeteners": (
        "candy", "candies", "chocolate", "cookie", "biscuit", "chip", "snack", "confection",
        "gum", "honey", "syrup", "sugar", "jelly", "jellies", "gummy", "gummies", "cake",
        "wafer",
    ),
    "Condiments": (
        "sauce", "ketchup", "mustard", "mayonnaise", "dressing", "relish", "pickle",
        "vinegar", "chutney", "salsa",
    ),
    "Herbs and spices": (
        "spice", "seasoning", "curry", "paprika", "cumin", "turmeric", "cinnamon", "herb",
        "masala", "oregano",
    ),
    "Beverages": (
        "juice", "drink", "beverage", "tea", "coffee", "water", "soda", "kombucha", "wine",
        "beer",
    ),
    "Infant products": ("infant", "baby", "formula"),
    "Egg products": ("egg",),
}

CONSUMER_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Toys and games": ("toy", "game", "doll", "puzzle", "plush", "rattle"),
    "Electronics": (
        "charger", "battery", "batteries", "power bank", "cable", "adapter", "headphone",
        "speaker", "laptop", "phone", "extension cord", "power bar",
    ),
    "Appliances": (
        "heater", "kettle", "toaster", "blender", "fan", "dehumidifier", "microwave", "oven",
        "stove", "dryer", "washer", "air fryer", "refrigerator",
    ),
    "Sports and fitness": (
        "bicycle", "bike", "helmet", "ski", "treadmill", "scooter", "skateboard", "fitness",
        "exercise", "kayak",
    ),
    "Clothing and accessories": (
        "sweater", "jacket", "hoodie", "shirt", "pants", "sleepwear", "pajama", "pyjama",
        "dress", "boot", "shoe", "hat", "mitten", "drawstring",
    ),
    "Furniture and furnishings": (
        "chair", "table", "dresser", "furniture", "mattress", "sofa", "shelf", "shelves",
        "cabinet",
    ),
    "Household items": ("blind", "candle", "lamp", "cookware", "crib", "stroller", "bassinet"),
    "Tools and accessories": ("drill", "saw", "nailer", "ladder", "tool", "generator", "mower"),
    "Beauty and personal care": ("shampoo", "lotion", "cosmetic", "razor", "sunscreen"),
    "Chemicals": ("cleaner", "detergent", "chemical", "paint", "solvent", "glue"),
}


def _compile(vocab: dict[str, tuple[str, ...]]) -> dict[str, re.Pattern[str]]:
    return {
        family: re.compile(
            r"\b(?:" + "|".join(re.escape(word) for word in words) + r")(?:s|es)?\b",
            re.IGNORECASE,
        )
        for family, words in vocab.items()
    }


_FOOD_PATTERNS = _compile(FOOD_FAMILY_KEYWORDS)
_CONSUMER_PATTERNS = _compile(CONSUMER_FAMILY_KEYWORDS)

# Leading hazard phrases, e.g. "Undeclared sesame in <product>".
_HAZARD_PREFIX = re.compile(
    r"^(?:undeclared|non-permitted|presence of|elevated|high levels? of|excess|possible|"
    r"extraneous)\b.*?\bin\b",
    re.IGNORECASE,
)
# Clauses that describe the hazard rather than the product.
_HAZARD_SUFFIX = re.compile(
    r"\b(?:recalled|may contain|may be|due to|because of|could)\b.*$", re.IGNORECASE
)
# "<Company> recalls <product>"
_COMPANY_RECALLS = re.compile(r"^.*?\brecalls\b", re.IGNORECASE)


def title_group_id(title: str) -> str:
    """Stable id for an exact (whitespace-normalized, case-sensitive) title."""
    return "tg_" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]


def add_title_groups(records: pd.DataFrame) -> pd.DataFrame:
    out = records.copy()
    out["title_group"] = out["title"].map(title_group_id)
    out["title_group_size"] = out.groupby("title_group")["id"].transform("size").astype(int)
    return out


def product_text(title: str, product: str | None) -> str:
    """Isolate the product description so hazard words (e.g. 'undeclared milk') don't match."""
    if isinstance(product, str) and product:
        return product
    text = _COMPANY_RECALLS.sub("", title) if _COMPANY_RECALLS.search(title) else title
    text = _HAZARD_PREFIX.sub("", text)
    text = _HAZARD_SUFFIX.sub("", text)
    return text.strip()


def assign_family(
    title: str, product: str | None, categories: list[str], domain: str
) -> dict[str, object]:
    specific = [c for c in categories if c not in GENERIC_CATEGORIES]
    if len(specific) == 1:
        return _family(specific[0], "official_category", "high", None)
    if len(specific) > 1:
        return _family(specific[0], "official_category", "medium", "multiple_official_categories")

    use_food = domain == "food" or "Food" in categories
    patterns = _FOOD_PATTERNS if use_food else _CONSUMER_PATTERNS
    text = product_text(title, product)
    matches = [family for family, pattern in patterns.items() if pattern.search(text)]

    if len(matches) == 1:
        return _family(matches[0], "keyword", "medium", "keyword_match_only")
    if len(matches) > 1:
        return _family(None, "keyword", "low", "ambiguous_keywords: " + " | ".join(matches))
    return _family(None, "none", "low", "no_keyword_match")


def _family(
    family: str | None, source: str, confidence: str, reason: str | None
) -> dict[str, object]:
    return {
        "product_family": family,
        "family_source": source,
        "family_confidence": confidence,
        "needs_review": confidence != "high",
        "review_reason": reason,
    }


def add_product_families(records: pd.DataFrame) -> pd.DataFrame:
    assigned = [
        assign_family(row.title, row.product, row.categories, row.domain)
        for row in records.itertuples(index=False)
    ]
    families = pd.DataFrame(assigned, index=records.index).astype(object)
    return pd.concat([records, families], axis=1)

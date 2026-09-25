"""The diagnosis's payment kinds: `streams._evidence` with its drops split into shop/fee, Decoy and Filler."""
from recurring_family.streams import _evidence, _words, _SHOP_WORDS, _DECOY_WORDS, HOME_MCC


def kind_of(d, m):
    k, fams, _ = _evidence(d, m)
    home = m in HOME_MCC
    if k in ("drop", "stray"):
        words = set(_words(d))
        if not words or words & _SHOP_WORDS:
            return "drop:shop/fee"
        if words & _DECOY_WORDS:
            return "drop:decoy"
        return "drop:filler-offhome"
    if k == "filler":
        return "filler-home" if home and len(fams) == 1 and HOME_MCC[m] in fams else "family-word-wrong-mcc"
    if k == "ambiguous":
        return "ambiguous"
    return "family"

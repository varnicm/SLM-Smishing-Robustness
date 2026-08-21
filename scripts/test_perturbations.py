"""Quick offline validation of the non-model transforms + URL invariant."""
import random
from perturbations import (
    mask_urls, restore_urls, character_level, word_level, SynonymSource,
)

# Real corpus samples the user pasted
SAMPLES = [
    "Notif: From: Usps CONTENT: A parcel needs to be delivered for you but is pending an update. To update your preferences head to 83hm2uaj.kzra.in/?/OVA013 Regards, USPS. Shipment#JYTHII1Z8381N4G4",
    "See the intimate pictures Kelly released 4 you! Unlock below...http://dtwxzf.com/003u52r",
    "WELLS FARG O: Account temporarily locked. Visit use.app.link/wf",
    "Verizon You've won a prize! Go to bit.ly/yourprize001 to claim your $500 Amazon gift card",
    "Welcome to join our BtC discussion Study Group, click whatsapp://chat/?code=15Ro5nXJ3Eml36gE3JZk7Taf",
]

print("=" * 70)
print("TEST 1: URL masking round-trips exactly")
print("=" * 70)
for s in SAMPLES:
    masked, urls = mask_urls(s)
    restored = restore_urls(masked, urls)
    ok = restored == s
    print(f"[{'OK ' if ok else 'FAIL'}] urls_found={len(urls)}  {urls}")
    assert ok, f"round-trip failed:\n {s}\n {restored}"

print()
print("=" * 70)
print("TEST 2: character-level never alters URLs")
print("=" * 70)
rng = random.Random(0)
for s in SAMPLES:
    _, urls = mask_urls(s)
    out, edits = character_level(s, frac=0.3, rng=rng)
    # every original URL must still appear verbatim in the output
    all_present = all(u in out for u in urls)
    print(f"[{'OK ' if all_present else 'FAIL'}] edits={len(edits):2d}  URLs_intact={all_present}")
    print(f"        {out}")
    assert all_present, f"URL mangled!\n {out}"

print()
print("=" * 70)
print("TEST 3: word-level substitution + URL preservation")
print("=" * 70)
# tiny fake counter-fitted lookup to exercise the path deterministically
fake = SynonymSource(lookup={
    "account": ["profile"], "prize": ["reward"], "won": ["earned"],
    "locked": ["frozen"], "claim": ["collect"], "gift": ["voucher"],
    "parcel": ["package"], "update": ["revise"], "pictures": ["photos"],
})
rng = random.Random(1)
for s in SAMPLES:
    _, urls = mask_urls(s)
    out, edits = word_level(s, frac=0.3, rng=rng, syn=fake)
    all_present = all(u in out for u in urls)
    print(f"[{'OK ' if all_present else 'FAIL'}] subs={len(edits):2d}  URLs_intact={all_present}")
    for e in edits:
        print(f"        {e.original!r} -> {e.replacement!r}")
    assert all_present

print("\nALL TESTS PASSED")

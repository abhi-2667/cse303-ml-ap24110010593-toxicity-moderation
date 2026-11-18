"""
download_data.py
----------------
Step 2 of the project: get the Jigsaw Toxic Comment dataset into data/raw/.

1. Tries the official Kaggle API (needs a Kaggle account + API token and
   you must accept the competition rules once on the Kaggle website).
2. If that fails, downloads the same train.csv from a public GitHub mirror
   (the dataset is CC0, so this is allowed) and checks it has 159,571 rows.
3. If both fail, prints setup instructions and writes a realistic 2,000-row
   SYNTHETIC fallback file: data/raw/train_sample.csv so the rest of the
   pipeline still runs end-to-end.

Run:  python download_data.py
"""
import random
import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "jigsaw-toxic-comment-classification-challenge"
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

SETUP_INSTRUCTIONS = """
=====================  KAGGLE SETUP INSTRUCTIONS  =====================
1. Create a free account at https://www.kaggle.com
2. Open Account settings -> 'API' -> 'Create New Token'
   (this downloads kaggle.json)
3. Put kaggle.json here and restrict permissions:
     Linux / Mac : ~/.kaggle/kaggle.json   (chmod 600 ~/.kaggle/kaggle.json)
     Windows     : C:\\Users\\<you>\\.kaggle\\kaggle.json
   (Or set env vars KAGGLE_USERNAME and KAGGLE_KEY.)
4. Visit https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules
   and click 'I Understand and Accept' (required once, otherwise 403 error).
5. Re-run:  python download_data.py
=======================================================================
"""


def unzip_all(folder: Path) -> None:
    """Extract every .zip in the folder (the competition zip contains more zips)."""
    for _ in range(3):  # nested zips: outer zip -> train.csv.zip -> train.csv
        zips = list(folder.glob("*.zip"))
        if not zips:
            break
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(folder)
            z.unlink()


def try_kaggle_download() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"Downloading '{COMPETITION}' from Kaggle ...")
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
        unzip_all(RAW_DIR)
        return (RAW_DIR / "train.csv").exists()
    except (Exception, SystemExit) as err:  # no credentials (kaggle exits), rules not accepted, no internet...
        print(f"Kaggle download failed: {type(err).__name__}: {err}")
        return False


MIRROR_URL = "https://raw.githubusercontent.com/iampukar/toxic-comments-classification/master/train.csv"


def try_github_mirror() -> bool:
    """Download train.csv from a public GitHub copy and sanity-check it."""
    try:
        import urllib.request

        print("Trying GitHub mirror ...")
        target = RAW_DIR / "train.csv"
        urllib.request.urlretrieve(MIRROR_URL, target)
        df = pd.read_csv(target)
        ok = df.shape[0] == 159571 and all(c in df.columns for c in ["comment_text"] + LABELS)
        if not ok:
            print(f"Downloaded file looks wrong (shape {df.shape}) - ignoring it.")
            target.unlink()
        return ok
    except Exception as err:
        print(f"GitHub mirror failed: {type(err).__name__}: {err}")
        return False


# --------------------------------------------------------------------------
# Fallback: realistic synthetic sample (2,000 rows, same columns as train.csv)
# --------------------------------------------------------------------------
TOPICS = ["the history section", "the infobox", "the references", "the lead paragraph",
          "the new policy", "the release schedule", "the photo captions", "the sports results",
          "the music category", "the science section", "the biography", "the city page"]

CLEAN_TEMPLATES = [
    "Thanks for the edit, {t} reads much better now.",
    "Can someone please add a reliable source for {t}?",
    "I think {t} makes a fair point, but it needs more detail.",
    "Great work on {t}, very informative and well written.",
    "I disagree with {t}, but I respect the effort that went into it.",
    "Could we discuss {t} on the talk page before changing it again?",
    "I fixed a few typos in {t} and added two citations.",
    "Welcome to the project! Please read the guidelines about {t}.",
    "The discussion about {t} was helpful, thank you everyone.",
    "Does anyone know when {t} was last updated?",
    "You are right about {t}, I will update it tonight.",
    "You are welcome to improve {t} if you have better sources.",
    "I think you are missing a citation in {t}, but it is a good start.",
    "You are doing a great job on {t}, keep going.",
    "Please do not call other editors names, it breaks the civility rules.",
    "Calling someone an idiot or a moron on {t} is against policy, please stay civil.",
    "Damn, this is a great article, especially {t}!",
    "The caption in {t} says 'clown car' but it should say 'minivan'.",
    "Someone left trash and graffiti in the photo from {t}, can we crop it?",
    "Honestly the hell of it is that {t} was correct all along.",
    "You are a {p} person, thanks for sorting out {t}.",
    "You are a {p} editor.", "You are very {p}, thank you for the help!",
    "Thank you, you are a lifesaver.", "Are you the person who rewrote {t}?",
    "Your edits on {t} are really appreciated.", "You are a {p} contributor to this project.",
    "You are welcome to edit this page.", "Hope you are doing well, thanks for checking {t}.",
]

POSITIVE = ["wonderful", "kind", "helpful", "brilliant", "great", "amazing", "thoughtful", "patient"]

INSULTS = ["idiot", "moron", "loser", "stupid", "pathetic", "clown", "imbecile", "trash"]
INSULT_TEMPLATES = [
    "You are such an {w}.", "What a {w} you are.", "Only an {w} would write this.",
    "Stop editing, you {w}.", "Everyone here thinks you are a {w}.",
    "Go away, {w}.", "Nobody cares about your edits, {w}.", "Typical {w} behaviour from this user.",
    "Another {w} ruining the page again.", "Learn to read, {w}.", "That editor is a total {w}.",
]
OBSCENE_TEMPLATES = [
    "This is complete bullshit.", "What the hell is wrong with this crap?",
    "Shut the hell up, nobody asked for your damn opinion.", "This page is a steaming pile of crap.",
    "Quit this bullshit and get a life.", "Damn, what a crappy article.",
    "Who the hell wrote this garbage section?", "This whole thing is crap and you know it.",
]
SEVERE_TEMPLATES = [
    "YOU ABSOLUTE WORTHLESS DISGUSTING WASTE OF SPACE, I DESPISE YOU AND EVERYTHING YOU WRITE.",
    "You are the most vile, worthless garbage on this entire site and everybody hates you.",
    "I hope you fail at everything, you disgusting worthless creature.",
]
THREAT_TEMPLATES = [
    "I will find you and make you regret this.", "Watch your back, I know where you live.",
    "Stop or I will hurt you, this is your last warning.", "I am going to come after you, just wait.",
    "You will pay for this, I promise you.",
]
HATE_TEMPLATES = [
    "People like you from that country ruin everything, go back where you came from.",
    "I can't stand your kind, you do not belong on this site.",
    "All of them from that religion are liars and should be banned.",
    "Your whole community is a disgrace and unwelcome here.",
]

# Leetspeak / symbol obfuscations of insult words
# NOTE: "1d10t", "i.d.i.o.t", "st00pid" and "m0r0n" are deliberately NOT used here, so the
# notebook's robustness test uses spellings the model has never seen in training.
OBFUSCATE = {
    "idiot": ["!d!ot", "id1ot", "1diot", "idi0t"],
    "moron": ["mor0n", "m.o.r.o.n", "m0ron"],
    "loser": ["l0s3r", "l.o.s.e.r", "lo$er"],
    "stupid": ["stup1d", "s.t.u.p.i.d", "stupld"],
    "pathetic": ["p4th3t1c", "pathet1c"],
    "clown": ["cl0wn", "c.l.o.w.n"],
    "imbecile": ["imb3c1le", "1mbecile"],
    "trash": ["tr@sh", "tr4sh", "t.r.a.s.h"],
}


def make_sample(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    def clean_sentence():
        return rng.choice(CLEAN_TEMPLATES).format(t=rng.choice(TOPICS), p=rng.choice(POSITIVE))

    def build(profile):
        parts = []
        if "insult" in profile:
            w = rng.choice(INSULTS)
            if rng.random() < 0.25:               # ~25% of insults are obfuscated
                w = rng.choice(OBFUSCATE[w])
            parts.append(rng.choice(INSULT_TEMPLATES).format(w=w))
        if "obscene" in profile:
            parts.append(rng.choice(OBSCENE_TEMPLATES))
        if "severe_toxic" in profile:
            parts.append(rng.choice(SEVERE_TEMPLATES))
        if "threat" in profile:
            parts.append(rng.choice(THREAT_TEMPLATES))
        if "identity_hate" in profile:
            parts.append(rng.choice(HATE_TEMPLATES))
        for _ in range(rng.choice([0, 1, 1, 2])):  # most toxic comments also contain harmless text
            parts.append(clean_sentence())
        rng.shuffle(parts)
        text = " ".join(parts)
        return text.upper() if rng.random() < 0.08 else text

    # (set of active labels besides 'toxic', how many rows)
    profiles = [
        (("insult",), 170), (("obscene",), 90), (("obscene", "insult"), 110),
        (("threat",), 40), (("threat", "insult"), 20),
        (("severe_toxic", "obscene", "insult"), 40),
        (("identity_hate",), 25), (("identity_hate", "insult"), 25),
    ]
    rows = []
    for profile, count in profiles:
        for _ in range(count):
            rec = {lab: 0 for lab in LABELS}
            rec["toxic"] = 1                      # any toxic sub-type is also 'toxic'
            for lab in profile:
                rec[lab] = 1
            rec["comment_text"] = build(profile)
            rows.append(rec)
    while len(rows) < n_rows:                     # the rest are clean comments
        rec = {lab: 0 for lab in LABELS}
        rec["comment_text"] = " ".join(clean_sentence() for _ in range(rng.randint(1, 3)))
        rows.append(rec)

    # Label noise (real Jigsaw labels are noisy too): ~3% of toxic rows labelled clean,
    # ~1.5% of clean rows labelled toxic.
    for rec in rows:
        if rec["toxic"] == 1 and rng.random() < 0.03:
            for lab in LABELS:
                rec[lab] = 0
        elif rec["toxic"] == 0 and rng.random() < 0.015:
            rec["toxic"] = 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", [f"sample_{i:05d}" for i in range(len(df))])
    return df[["id", "comment_text"] + LABELS]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "train.csv").exists():
        print("data/raw/train.csv already exists - nothing to do.")
        return
    if try_kaggle_download():
        print("Done! Real dataset is in data/raw/ (train.csv, test.csv, ...).")
        return
    if try_github_mirror():
        print("Done! Real train.csv (159,571 rows) is in data/raw/ (from the GitHub mirror).")
        return
    print(SETUP_INSTRUCTIONS)
    sample_path = RAW_DIR / "train_sample.csv"
    make_sample().to_csv(sample_path, index=False)
    print(f"Wrote SYNTHETIC fallback sample ({sample_path}). Replace it with the real "
          "data later by following the steps above.")


if __name__ == "__main__":
    main()

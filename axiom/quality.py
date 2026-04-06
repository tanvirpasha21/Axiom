"""
AXIOM Quality Filter
====================
Classifies papers as Q1, Q2, Q3, Q4, or top_conference based on their
publication venue. Uses SCImago / JCR quartile data for journals and
ERA / CORE rankings for conferences.

The filter is intentionally conservative:
  - Unknown venues get no rank (None) — not filtered out, just unranked
  - Only ingest to warehouse Q1/Q2 papers when quality_filter=True
  - top_conference is treated as equivalent to Q1 in all downstream logic

Usage:
    from axiom.quality import classify_venue, is_high_quality

    rank = classify_venue("IEEE Transactions on Pattern Analysis")
    # → "Q1"

    ok = is_high_quality(paper)
    # → True  if Q1, Q2, or top_conference
"""

from __future__ import annotations
import re
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Q1 JOURNALS  (SCImago Q1 / JCR top quartile)
# ──────────────────────────────────────────────────────────────────────────────
_Q1_JOURNALS: list[str] = [
    # IEEE flagship journals
    "ieee transactions on pattern analysis and machine intelligence",
    "ieee transactions on neural networks and learning systems",
    "ieee transactions on knowledge and data engineering",
    "ieee transactions on image processing",
    "ieee transactions on cybernetics",
    "ieee transactions on industrial informatics",
    "ieee transactions on information theory",
    "ieee transactions on signal processing",
    "ieee transactions on communications",
    "ieee transactions on mobile computing",
    "ieee transactions on software engineering",
    "ieee transactions on dependable and secure computing",
    "ieee transactions on parallel and distributed systems",
    "ieee transactions on computers",
    "ieee transactions on automatic control",
    "ieee transactions on robotics",
    "ieee transactions on medical imaging",
    "ieee transactions on geoscience and remote sensing",
    "ieee journal of selected areas in communications",
    "ieee wireless communications",
    "ieee network",
    "proceedings of the ieee",

    # ACM flagship journals
    "journal of the acm",
    "acm computing surveys",
    "acm transactions on information systems",
    "acm transactions on database systems",
    "acm transactions on computer systems",
    "acm transactions on graphics",
    "acm transactions on intelligent systems and technology",
    "acm transactions on knowledge discovery from data",
    "communications of the acm",

    # Machine learning / AI
    "journal of machine learning research",
    "machine learning",
    "artificial intelligence",
    "international journal of computer vision",
    "neural networks",
    "neural computation",
    "pattern recognition",
    "data mining and knowledge discovery",
    "nature machine intelligence",
    "nature communications",
    "nature",
    "science",
    "science advances",
    "plos computational biology",

    # NLP
    "computational linguistics",
    "transactions of the association for computational linguistics",

    # Databases / information systems
    "vldb journal",
    "information systems",
    "the vldb journal",

    # Security
    "ieee transactions on information forensics and security",
    "computers & security",
    "journal of cybersecurity",

    # Multidisciplinary
    "expert systems with applications",
    "information fusion",
    "knowledge-based systems",       # borderline Q1/Q2 but included
    "information sciences",
]

# ──────────────────────────────────────────────────────────────────────────────
# Q2 JOURNALS
# ──────────────────────────────────────────────────────────────────────────────
_Q2_JOURNALS: list[str] = [
    "neurocomputing",
    "applied soft computing",
    "future generation computer systems",
    "journal of artificial intelligence research",
    "neural computing and applications",
    "applied intelligence",
    "engineering applications of artificial intelligence",
    "computer vision and image understanding",
    "journal of intelligent & fuzzy systems",
    "swarm and evolutionary computation",
    "cognitive computation",
    "international journal of machine learning and cybernetics",
    "soft computing",
    "evolutionary computation",
    "international journal of neural systems",
    "journal of network and computer applications",
    "computers & electrical engineering",
    "digital signal processing",
    "signal processing",
    "signal processing: image communication",
    "pattern recognition letters",
    "image and vision computing",
    "computer methods and programs in biomedicine",
    "biomedical signal processing and control",
    "ieee access",               # technically mega-journal but widely accepted Q2
    "peerj computer science",
    "frontiers in artificial intelligence",
    "iet image processing",
    "iet computer vision",
    "iet signal processing",
    "journal of the franklin institute",
    "automatica",
    "control engineering practice",
    "robotics and autonomous systems",
    "international journal of robotics research",
    "autonomous robots",
    "journal of systems and software",
    "software: practice and experience",
    "empirical software engineering",
    "information and software technology",
    "journal of web semantics",
    "semantic web",
    "social networks",
    "physica a: statistical mechanics and its applications",
    "chaos, solitons & fractals",
    "entropy",
    "mathematics",
    "algorithms",
    "sensors",
]

# ──────────────────────────────────────────────────────────────────────────────
# TOP CONFERENCES  (ERA A* / CORE A* — treated as Q1 equivalent)
# ──────────────────────────────────────────────────────────────────────────────
_TOP_CONFERENCES: list[str] = [
    # Machine Learning & AI
    "neurips", "nips", "advances in neural information processing",
    "icml", "international conference on machine learning",
    "iclr", "international conference on learning representations",
    "aaai", "association for the advancement of artificial intelligence",
    "ijcai", "international joint conference on artificial intelligence",
    "aistats", "artificial intelligence and statistics",
    "uai", "uncertainty in artificial intelligence",
    "colt", "computational learning theory",
    "ecml", "european conference on machine learning",

    # Computer Vision
    "cvpr", "conference on computer vision and pattern recognition",
    "iccv", "international conference on computer vision",
    "eccv", "european conference on computer vision",
    "bmvc", "british machine vision conference",
    "wacv",

    # NLP
    "acl", "association for computational linguistics",
    "emnlp", "empirical methods in natural language processing",
    "naacl", "north american chapter of the association for computational linguistics",
    "coling", "international conference on computational linguistics",
    "eacl",

    # Data Mining & Information Retrieval
    "kdd", "knowledge discovery and data mining",
    "sigir", "special interest group on information retrieval",
    "www", "world wide web conference",
    "wsdm", "web search and data mining",
    "icdm", "ieee international conference on data mining",
    "sdm", "siam international conference on data mining",
    "cikm", "conference on information and knowledge management",
    "recsys",

    # Databases
    "sigmod", "acm sigmod",
    "vldb", "very large data bases",
    "icde", "ieee international conference on data engineering",
    "pods",

    # Systems & Networks
    "sosp", "symposium on operating systems principles",
    "osdi", "operating systems design and implementation",
    "nsdi", "networked systems design and implementation",
    "usenix atc",
    "eurosys",
    "asplos",
    "isca",
    "micro", "ieee/acm international symposium on microarchitecture",
    "hpca",
    "sc", "supercomputing",
    "ipdps",
    "ppopp",

    # Security
    "ccs", "acm conference on computer and communications security",
    "ieee security and privacy", "ieee s&p", "sp",
    "usenix security",
    "ndss", "network and distributed system security",

    # Theory
    "stoc", "symposium on theory of computing",
    "focs", "foundations of computer science",
    "soda", "symposium on discrete algorithms",

    # Robotics / HCI / SE
    "icra", "ieee international conference on robotics and automation",
    "iros",
    "chi", "acm conference on human factors in computing systems",
    "uist",
    "icse", "international conference on software engineering",
    "fse", "foundations of software engineering",
    "ase", "automated software engineering",
    "issta",
    "cav",
]

# ──────────────────────────────────────────────────────────────────────────────
# Pre-compiled lookup sets for O(1) substring matching
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for matching."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_Q1_NORMALISED      = [_normalise(v) for v in _Q1_JOURNALS]
_Q2_NORMALISED      = [_normalise(v) for v in _Q2_JOURNALS]
_CONF_NORMALISED    = [_normalise(v) for v in _TOP_CONFERENCES]


def classify_venue(venue: str | None) -> Optional[str]:
    """
    Classify a venue string into Q1/Q2/top_conference or None.

    Returns:
        "Q1"              — SCImago/JCR Q1 journal
        "Q2"              — SCImago/JCR Q2 journal
        "top_conference"  — ERA A* / CORE A* conference (treated as Q1)
        None              — Unknown / unclassified venue
    """
    if not venue:
        return None

    norm = _normalise(venue)

    # Q1 journals checked FIRST to avoid conference acronyms shadowing journal names
    for j in _Q1_NORMALISED:
        if j in norm or norm in j:
            return "Q1"

    # Q2 journals
    for j in _Q2_NORMALISED:
        if j in norm or norm in j:
            return "Q2"

    # Conferences — use word-boundary matching for short acronyms (≤6 chars)
    # to avoid "sp" matching "special", "acl" matching "oracle", etc.
    for c in _CONF_NORMALISED:
        if len(c) <= 5:
            # Require whole-word match for very short acronyms
            if re.search(rf"\b{re.escape(c)}\b", norm):
                return "top_conference"
        else:
            if c in norm or norm in c:
                return "top_conference"

    return None


def is_high_quality(paper) -> bool:
    """
    Return True if the paper is in a Q1, Q2, or top_conference venue.
    Papers with no quality rank are treated as not high-quality.
    """
    return paper.quality_rank in ("Q1", "Q2", "top_conference")


def enrich_quality(paper):
    """
    Set paper.quality_rank in-place based on its venue.
    Returns the paper for chaining.
    """
    paper.quality_rank = classify_venue(paper.venue)
    return paper
